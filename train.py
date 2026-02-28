import collections
import json
import numpy as np
from pathlib import Path
import torch
from transformers import (
    BertTokenizerFast,
    BertForQuestionAnswering,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

MODEL_NAME = "bert-base-chinese"


def load_data(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_context(context_path):
    """載入段落陣列，索引即為段落 ID"""
    with open(context_path, encoding="utf-8") as f:
        contexts = json.load(f)
    return contexts


def prepare_train_features(examples, contexts, tokenizer, max_length=512):
    """處理訓練資料"""
    all_encodings = {
        "input_ids": [],
        "attention_mask": [],
        "start_positions": [],
        "end_positions": [],
    }
    
    skipped = 0
    
    for example in examples:
        question = example["question"]
        relevant_id = example["relevant"]
        answer = example["answers"][0]
        answer_text = answer["text"]
        answer_start = answer["start"]
        
        # 取得相關段落（直接用 ID 作為索引）
        if relevant_id >= len(contexts):
            print(f"Warning: Context ID {relevant_id} out of range")
            skipped += 1
            continue
        
        context = contexts[relevant_id]
        
        # 驗證答案在段落中
        if answer_text not in context:
            print(f"Warning: Answer '{answer_text}' not in context {relevant_id}")
            skipped += 1
            continue
        
        # 驗證 answer_start 位置
        if context[answer_start:answer_start + len(answer_text)] != answer_text:
            # 如果 start 位置不對，用文字搜尋
            try:
                answer_start = context.index(answer_text)
            except ValueError:
                print(f"Warning: Cannot find answer '{answer_text}' in context {relevant_id}")
                skipped += 1
                continue
        
        answer_end = answer_start + len(answer_text)
        
        # Tokenize
        encoding = tokenizer(
            question,
            context,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_offsets_mapping=True,
        )
        
        offsets = encoding["offset_mapping"]
        sequence_ids = encoding.sequence_ids()
        
        # 找 context 部分的 token 範圍
        context_start_idx = 0
        while context_start_idx < len(sequence_ids) and sequence_ids[context_start_idx] != 1:
            context_start_idx += 1
        
        if context_start_idx >= len(sequence_ids):
            skipped += 1
            continue
        
        context_end_idx = context_start_idx
        while context_end_idx < len(sequence_ids) and sequence_ids[context_end_idx] == 1:
            context_end_idx += 1
        context_end_idx -= 1
        
        # 找答案對應的 token
        token_start = None
        token_end = None
        
        for idx in range(context_start_idx, context_end_idx + 1):
            if offsets[idx] is None:
                continue
            
            # token 的字元範圍
            token_char_start, token_char_end = offsets[idx]
            
            # 找 start token
            if token_start is None and token_char_start <= answer_start < token_char_end:
                token_start = idx
            
            # 找 end token
            if token_char_start < answer_end <= token_char_end:
                token_end = idx
                break
        
        # 如果答案被截斷，跳過
        if token_start is None or token_end is None or token_end < token_start:
            skipped += 1
            continue
        
        all_encodings["input_ids"].append(encoding["input_ids"])
        all_encodings["attention_mask"].append(encoding["attention_mask"])
        all_encodings["start_positions"].append(token_start)
        all_encodings["end_positions"].append(token_end)
    
    print(f"✓ Prepared {len(all_encodings['input_ids'])} examples, skipped {skipped}")
    return all_encodings


class QADataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}


def train(train_path, context_path, model_dir):
    print("Loading tokenizer and model...")
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    model = BertForQuestionAnswering.from_pretrained(MODEL_NAME)

    print("Loading data...")
    data = load_data(train_path)
    contexts = load_context(context_path)

    # 分割訓練集和驗證集 (90% train, 10% val)
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    eval_data = data[split_idx:]
    
    print(f"Loaded {len(train_data)} training examples and {len(eval_data)} validation examples")
    
    print("Preparing training features...")
    train_encodings = prepare_train_features(train_data, contexts, tokenizer)
    train_dataset = QADataset(train_encodings)
    
    print("Preparing validation features...")
    eval_encodings = prepare_train_features(eval_data, contexts, tokenizer)
    eval_dataset = QADataset(eval_encodings)

    # 保存驗證集數據（訓練後評估用）
    print("Saving validation data...")
    with open(f"{model_dir}/val_data.json", "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)

    print("Starting training...")
    args = TrainingArguments(
        output_dir=model_dir,
        per_device_train_batch_size=4,
        num_train_epochs=2,
        learning_rate=2e-5,
        logging_steps=100,
        save_steps=500,
        save_total_limit=10,
        warmup_steps=500,
        weight_decay=0.01,
        fp16=True,
        # 加這三行
        evaluation_strategy="steps",  # 或 "epoch"
        eval_steps=500,  # 每 500 steps 評估一次
        load_best_model_at_end=True,  # 訓練結束載入最佳模型
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model, 
        args=args, 
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()

    print("Saving model...")
    trainer.save_model(model_dir)   # ✅ 這個會存 trainer.model（best）
    tokenizer.save_pretrained(model_dir)
    
    # 訓練完成後，對驗證集進行完整評估
    print("\n" + "="*60)
    print("Evaluating on validation set...")
    print("="*60)
    
    val_pred_path = f"{model_dir}/val_predictions.json"
    predict_for_validation(model_dir, context_path, eval_data, contexts, val_pred_path)
    
    # 計算 EM 和 F1
    print("\nComputing metrics...")
    predictions = load_data(val_pred_path)
    
    em_scores = []
    f1_scores = []
    
    for example in eval_data:
        qid = example["id"]
        true_answer = example["answers"][0]["text"]
        pred_answer = predictions.get(qid, "")
        
        em_scores.append(compute_em(pred_answer, true_answer))
        f1_scores.append(compute_f1(pred_answer, true_answer))
    
    val_em = np.mean(em_scores)
    val_f1 = np.mean(f1_scores)
    
    print("\n" + "="*60)
    print(f"📊 Validation Results:")
    print(f"   EM:  {val_em:.4f} ({val_em*100:.2f}%)")
    print(f"   F1:  {val_f1:.4f} ({val_f1*100:.2f}%)")
    print("="*60)
    
    # 保存評估結果
    with open(f"{model_dir}/val_results.json", "w", encoding="utf-8") as f:
        json.dump({"em": val_em, "f1": val_f1, "count": len(eval_data)}, f, indent=2)
    
    print("✓ Training complete!")


def predict_for_validation(model_dir, context_path, eval_data, contexts, pred_path):
    """專門用於驗證集的預測（eval_data 已經是列表格式）"""
    print("Loading model for validation...")
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    model = BertForQuestionAnswering.from_pretrained(model_dir)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Predicting {len(eval_data)} validation examples...")
    predictions = {}
    
    for i, qa in enumerate(eval_data):
        if (i + 1) % 100 == 0:
            print(f"Progress: {i + 1}/{len(eval_data)}")
        
        qid = qa["id"]
        question = qa["question"]
        
        # 驗證集只需要預測相關段落（因為已知正確段落）
        relevant_id = qa["relevant"]
        context = contexts[relevant_id]
        
        inputs = tokenizer(
            question, 
            context, 
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        start_scores = outputs.start_logits[0]  # (n,)
        end_scores   = outputs.end_logits[0]    # (n,)
        n = start_scores.size(0)

        max_answer_length = 80

        token_type_ids = inputs.get("token_type_ids", None)
        if token_type_ids is not None:
            tt = token_type_ids[0]  # 0=question, 1=context
            start_scores = start_scores.masked_fill(tt != 1, float("-inf"))
            end_scores   = end_scores.masked_fill(tt != 1, float("-inf"))

        score_matrix = start_scores[:, None] + end_scores[None, :]  # (n, n)

        # 1) end >= start
        mask_triu = torch.triu(torch.ones(n, n, dtype=torch.bool, device=score_matrix.device))

        # 2) end - start + 1 <= max_answer_length
        idx = torch.arange(n, device=score_matrix.device)
        mask_len = (idx[None, :] - idx[:, None]) < max_answer_length  # (n,n) True if j-i < L

        mask = mask_triu & mask_len
        score_matrix = score_matrix.masked_fill(~mask, float("-inf"))

        best_idx = score_matrix.argmax()
        best_start = (best_idx // n).item()
        best_end   = (best_idx % n).item()

        answer_ids = inputs["input_ids"][0][best_start:best_end + 1]
        answer = tokenizer.decode(answer_ids, skip_special_tokens=True).replace(" ", "")
        predictions[qid] = answer
    
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Validation predictions saved to {pred_path}")


def compute_em(pred, truth):
    """計算 Exact Match"""
    return int(pred == truth)


def compute_f1(pred, truth):
    """計算 F1 Score"""
    pred_tokens = list(pred)
    truth_tokens = list(truth)
    
    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


def predict(model_dir, context_path, data_path, pred_path):
    """預測階段：對每個問題的所有段落評分，選最高的"""
    print("Loading model...")
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    model = BertForQuestionAnswering.from_pretrained(model_dir)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    
    print("Loading data...")
    data = load_data(data_path)
    contexts = load_context(context_path)
    
    print(f"Predicting {len(data)} examples...")
    predictions = {}
    
    for i, qa in enumerate(data):
        if (i + 1) % 100 == 0:
            print(f"Progress: {i + 1}/{len(data)}")
        
        qid = qa["id"]
        question = qa["question"]
        paragraph_ids = qa["paragraphs"]
        
        best_answer = ""
        best_score = float('-inf')
        
        # 對每個段落都評分
        for pid in paragraph_ids:
            if pid >= len(contexts):
                continue
            
            context = contexts[pid]
            
            inputs = tokenizer(
                question, 
                context, 
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            start_scores = outputs.start_logits[0]  # (n,)
            end_scores   = outputs.end_logits[0]    # (n,)
            n = start_scores.size(0)

            max_answer_length = 80  # 你可以調

            # 只允許從 context 取答案（避免抽到 question / special / pad）
            token_type_ids = inputs.get("token_type_ids", None)
            if token_type_ids is not None:
                tt = token_type_ids[0]  # 0=question, 1=context
                start_scores = start_scores.masked_fill(tt != 1, float("-inf"))
                end_scores   = end_scores.masked_fill(tt != 1, float("-inf"))

            # 全域最優 span：score(i,j)=start[i]+end[j]，限制 j>=i 且長度<=max_answer_length
            score_matrix = start_scores[:, None] + end_scores[None, :]  # (n, n)

            idx = torch.arange(n, device=score_matrix.device)
            mask_triu = torch.triu(torch.ones(n, n, dtype=torch.bool, device=score_matrix.device))  # j>=i
            mask_len  = (idx[None, :] - idx[:, None]) < max_answer_length  # j-i < L
            mask = mask_triu & mask_len

            score_matrix = score_matrix.masked_fill(~mask, float("-inf"))

            best_idx = score_matrix.argmax()
            start_idx = (best_idx // n).item()
            end_idx   = (best_idx % n).item()

            score = score_matrix.view(-1)[best_idx]  # 這就是該段落的最佳分數

            if score > best_score:
                best_score = score
                answer_ids = inputs["input_ids"][0][start_idx:end_idx + 1]
                answer = tokenizer.decode(answer_ids, skip_special_tokens=True).replace(" ", "")
                best_answer = answer
                        
        # 如果答案為空，用問題中的關鍵字作為預設答案
        if not best_answer or best_answer.strip() == "":
            best_answer = "無答案"  # 給一個預設值
        
        predictions[qid] = best_answer
    
    print("Saving predictions...")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Predictions saved to {pred_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Train:   python train.py train <train.json> <context.json> <model_dir>")
        print("  Predict: python train.py predict <model_dir> <context.json> <test.json> <output.json>")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "train":
        if len(sys.argv) != 5:
            print("Usage: python train.py train <train.json> <context.json> <model_dir>")
            sys.exit(1)
        train(sys.argv[2], sys.argv[3], sys.argv[4])
    elif mode == "predict":
        if len(sys.argv) != 6:
            print("Usage: python train.py predict <model_dir> <context.json> <test.json> <output.json>")
            sys.exit(1)
        predict(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        print(f"Unknown mode: {mode}")
        print("Use 'train' or 'predict'")
        sys.exit(1)
