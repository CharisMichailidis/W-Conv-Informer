

python -u main_informer.py --model informer --data BTCDateInformerNews --features S  --freq d --seq_len 30 --label_len 20 --pred_len 10 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5



python -u main_informer.py --model informer --data BTCDateInformerNews --features S  --freq d --seq_len 30 --label_len 20 --pred_len 10 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 1


python -u main_informer.py --model informer --data BTCDateInformerNews --features MS  --freq d --seq_len 30 --label_len 20 --pred_len 10 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5



python -u main_informer.py --model informer --data BTCDateInformerNews --features S  --freq d --seq_len 100 --label_len 50 --pred_len 30 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5



python -u main_informer.py --model informer --data BTCDateInformerNews --features MS  --freq d --seq_len 100 --label_len 50 --pred_len 30 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5


# Experiments   pred_len = 60 , seq_len = 100, label_len = 80 or label_len = 50


# Experiments



python -u main_informer.py --model informer --data BTCDateInformerNews --features S  --freq d --seq_len 50 --label_len 25 --pred_len 30 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5



python -u main_informer.py --model informer --data BTCDateInformerNews --features MS  --freq d --seq_len 50 --label_len 25 --pred_len 30 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5


# MS 1 itetration

python -u main_informer.py --model informer --data BTCDateInformerNews --features MS  --freq d --seq_len 50 --label_len 25 --pred_len 30 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 1



python -u main_informer.py --model informer --data BTCDateInformerNews --features S  --freq d --seq_len 60 --label_len 35 --pred_len 40 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5



python -u main_informer.py --model informer --data BTCDateInformerNews --features MS  --freq d --seq_len 60 --label_len 35 --pred_len 40 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5



