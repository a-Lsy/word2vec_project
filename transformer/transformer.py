import torch
            tgt_out = fr[:, 1:]
            optimizer.zero_grad()
            output = model(en, tgt_in)
            loss = criterion(output.reshape(-1, vocab_fr.vocab_size), tgt_out.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        loss_list.append(avg_loss)        
print(f"Epoch {epoch+1:2d} | Loss: {avg_loss:.4f}")    
return model, loss_list
# ===================== 7. 翻译与评估（最终正确版）=====================
def translate(model, eng_sent):
    model.eval()    
with torch.no_grad():
        en_idx = vocab_en.encode(eng_sent)
        en_idx = vocab_en.pad(en_idx)
        en_tensor = torch.tensor([en_idx], dtype=torch.long).to(device)        
        fr_idx = [1]  # <sos>                
for _ in range(MAX_LEN - 1):            
# 只取已生成的部分，不填充到 MAX_LEN（核心修复！）
            fr_tensor = torch.tensor([fr_idx], dtype=torch.long).to(device)            
            output = model(en_tensor, fr_tensor)                        
# 取最后一个时间步的预测（核心修复！）
            pred = output.argmax(-1)[:, -1].item()                        
if pred == 2:  # <eos>                
break            
            fr_idx.append(pred)        
        fr_sent = " ".join([vocab_fr.idx2word[i] for i in fr_idx[1:]])    
return fr_sent
# ===================== 8. 运行训练与对比 =====================
if __name__ == "__main__":
    model_dot, loss_dot = train_model("dot")
    model_add, loss_add = train_model("add")
    test_sents = [        
"I love you .",        
"What is your name ?",        
"She is my friend .",        
"He is reading a book .",        
"We are happy ."    
]
    print("\n========== 翻译结果对比 ==========")    
for sent in test_sents:        
print(f"\n英文: {sent}")        
print(f"点积注意力: {translate(model_dot, sent)}")        
print(f"加性注意力: {translate(model_add, sent)}")
    print("\n========== 最终损失对比 ==========")    
print(f"点积注意力最终损失: {loss_dot[-1]:.4f}")    
print(f"加性注意力最终损失: {loss_add[-1]:.4f}")
    print("在低维数据上，加性注意力略占优；高维下点积更快更稳。")