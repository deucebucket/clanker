#!/usr/bin/env python3
"""Test the V2 force predictor model standalone after training completes."""
import torch, sys, os, re
sys.path.insert(0, '/var/home/deucebucket/ai-drive/clanker-lang')
from training.train_v2 import ClankerForcePredictor, ROLES, ROLE_TO_IDX, NUM_ROLES, CENTER, pendulum_forward
from transformers import GPT2Config, GPT2Tokenizer
from demo.pendulum_v2 import PendulumV2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint_dir = 'training/checkpoints/best_v2'

if not os.path.exists(f'{checkpoint_dir}/model.pt'):
    print("No best_v2 checkpoint yet — waiting for training to finish")
    sys.exit(1)

# Load model
config = torch.load(f'{checkpoint_dir}/config.pt', map_location=device, weights_only=False)
model = ClankerForcePredictor(config)
model.load_state_dict(torch.load(f'{checkpoint_dir}/model.pt', map_location=device, weights_only=False))
model.to(device)
model.eval()

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

engine = PendulumV2()

def predict_model(text):
    tokens = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=64).to(device)
    with torch.no_grad():
        role_logits, forces, coeffs = model(tokens['input_ids'], tokens['attention_mask'])
    pred_roles = role_logits.argmax(-1)
    sl = min(tokens['input_ids'].shape[1], forces.shape[1])
    vadug = pendulum_forward(forces[:,:sl], coeffs[:,:sl], pred_roles[:,:sl], sl)
    v, a, d, u, g = vadug[0].cpu().numpy()
    return int(v), int(a), int(d), int(u), int(g)

def predict_engine(text):
    r = engine.process_text(text)
    return r[0].v, r[0].a, r[0].d, r[0].u, r[0].g

print('=' * 70)
print('V2 FORCE PREDICTOR — STANDALONE MODEL TEST')
print('=' * 70)

tests = [
    ('I am happy', 'positive'),
    ('I am so happy today', 'positive'),
    ('This is amazing', 'positive'),
    ('I love you', 'positive'),
    ('Best day ever', 'positive'),
    ('I am sad', 'negative'),
    ('I hate this', 'negative'),
    ('This is terrible', 'negative'),
    ('I feel awful', 'negative'),
    ('I am not happy', 'negative'),
    ('I want to die', 'crisis'),
    ('I want to kill myself', 'crisis'),
    ('Nobody would miss me', 'crisis'),
    ('I dont want to exist', 'crisis'),
    ('I cant take this anymore', 'crisis'),
    ('Life isnt worth living', 'crisis'),
    ('I want to hurt myself', 'crisis'),
    ('I cant keep going', 'crisis'),
    ('The meeting is at three', 'neutral'),
    ('I went to the store', 'neutral'),
    ('The weather is okay', 'neutral'),
    ('Oh great another Monday', 'negative'),
    ('Im fine', 'negative'),
    ('Whatever Im fine', 'negative'),
    ('this is fucking amazing', 'positive'),
    ('this is fucking terrible', 'negative'),
    ('we need to talk', 'negative'),
    ('I still love her', 'positive'),
    ('my heart is broken', 'negative'),
    ('bruh im done', 'negative'),
]

model_correct = 0
engine_correct = 0
total = len(tests)

print(f'\n{"Text":40s} {"Exp":8s} {"Engine":>8s} {"Model":>8s} {"Eng":>4s} {"Mod":>4s}')
print('-' * 70)

for text, expected in tests:
    ev, ea, ed, eu, eg = predict_engine(text)
    mv, ma, md, mu, mg = predict_model(text)
    
    def check(v, exp):
        if exp == 'positive': return v > 135
        elif exp == 'negative': return v < 120
        elif exp == 'crisis': return v < 80
        elif exp == 'neutral': return 115 <= v <= 145
    
    e_ok = check(ev, expected)
    m_ok = check(mv, expected)
    engine_correct += e_ok
    model_correct += m_ok
    
    e_mark = 'OK' if e_ok else 'XX'
    m_mark = 'OK' if m_ok else 'XX'
    print(f'  {text:38s} {expected:8s} V={ev:3d}     V={mv:3d}    {e_mark:>3s} {m_mark:>3s}')

print(f'\n{"":40s} {"":8s} {"Engine":>8s} {"Model":>8s}')
print(f'{"ACCURACY":40s} {"":8s} {engine_correct}/{total} ({100*engine_correct/total:.0f}%)  {model_correct}/{total} ({100*model_correct/total:.0f}%)')
print(f'\nModel: 7.7M params, 30MB. Predicts per-word forces, fixed math computes VADUG.')

# Also send results to dossybox if tailscale available
import subprocess
try:
    result_text = []
    result_text.append("V2 FORCE PREDICTOR MODEL TEST RESULTS")
    result_text.append(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
    result_text.append(f"Engine: {engine_correct}/{total} ({100*engine_correct/total:.0f}%)")
    result_text.append(f"Model:  {model_correct}/{total} ({100*model_correct/total:.0f}%)")
    with open('/tmp/model_test_results.txt', 'w') as f:
        f.write('\n'.join(result_text))
    subprocess.run(['tailscale', 'file', 'cp', '/tmp/model_test_results.txt', 'dossybox:'], timeout=10)
except:
    pass
