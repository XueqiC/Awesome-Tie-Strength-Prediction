# Output file
output_file="greedy.txt"

# Clear the previous content of the output file if it exists
>> $output_file

echo "Running experiments for BA"

# Setting 1 (no threshold needed)
python greedy.py --dataset='ba' --setting=1 --model=GCN --n_emb=32 --n_hidden=1024 --n_out=16 >> $output_file

python greedy.py --dataset='ba' --setting=2 --model=GCN --thre=5 --n_emb=32 --n_hidden=1024 --n_out=16 >> $output_file
python greedy.py --dataset='ba' --setting=3 --model=GCN --thre=5 --n_emb=32 --n_hidden=1024 --n_out=16 >> $output_file

for setting in {4..7}
do
    python greedy.py --dataset='ba' --setting=$setting --model=GCN --thre=0.2 --n_emb=32 --n_hidden=1024 --n_out=16 >> $output_file

done

# # Constants
tw="2e7"

# Setting 1 (specific case without threshold)
python greedy.py --dataset='msg' --setting=1 --model=GCN --n_emb=32 --n_hidden=512 --n_out=32 >> $output_file

# Settings 2 (specific thresholds)
python greedy.py --dataset='msg' --setting=2 --model=GCN --thre=5 --tw=$tw --n_emb=32 --n_hidden=512 --n_out=32 >> $output_file
python greedy.py --dataset='msg' --setting=3 --model=GCN --thre=5 --tw=$tw --n_emb=32 --n_hidden=512 --n_out=32 >> $output_file

for setting in {4..7}
do
    python greedy.py --dataset='msg' --setting=$setting --model=GCN --thre=0.2 --tw=$tw --n_emb=32 --n_hidden=512 --n_out=32 >> $output_file
done

# python greedy.py --dataset='tw' --setting=1 --model=MLP --n_emb=64 --n_hidden=1024 --n_out=32 >> $output_file
# python greedy.py --dataset='tw' --setting=2 --model=MLP --thre=1 --n_emb=64 --n_hidden=1024 --n_out=32 >> $output_file
# python greedy.py --dataset='tw' --setting=3 --model=MLP --thre=1 --n_emb=64 --n_hidden=1024  --n_out=32  >> $output_file
# python greedy.py --dataset='tw' --setting=4 --model=MLP --thre=0.2 --n_emb=64 --n_hidden=1024  --n_out=32 >> $output_file
# python greedy.py --dataset='tw' --setting=5 --model=MLP --thre=0.2 --n_emb=64 --n_hidden=1024  --n_out=32  >> $output_file
# python greedy.py --dataset='tw' --setting=6 --model=MLP --thre=0.05 --n_emb=64 --n_hidden=1024  --n_out=32  >> $output_file
# python greedy.py --dataset='tw' --setting=7 --model=MLP --thre=0.05 --n_emb=64 --n_hidden=1024  --n_out=32  >> $output_file