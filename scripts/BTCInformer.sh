#   seq_len = 30   ,    label_len = 6 ,  pred_len = 1


python -u main_informer.py --model informer --data BTCDateInformer --features S  --freq d --seq_len 100 --label_len 50 --pred_len 30 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5


python -u main_informer.py --model informer --data BTCDateInformer --features S --freq d --seq_len 50 --label_len 25 --pred_len 30 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5 
 

python -u main_informer.py --model informer --data BTCDateInformer --features S  --freq d --seq_len 30 --label_len 30 --pred_len 1 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5


python -u main_informer.py --model informer --data BTCDateInformer --features S  --freq d --seq_len 31 --label_len 1 --pred_len 30 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5