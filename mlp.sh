# Output file
# output_file="./final_result/mlp_future.txt"
output_file="mlp.txt"

# Clear the previous content of the output file if it exists
>> $output_file

# python main.py --dataset='ba' --setting=1 --model=MLP --n_emb=32 --n_hidden=1024 --n_out=16 >> $output_file

# python main.py --dataset='ba' --setting=2 --model=MLP --thre=5 --n_emb=32 --n_hidden=1024 --n_out=16 >> $output_file
# python main.py --dataset='ba' --setting=3 --model=MLP --thre=5 --n_emb=32 --n_hidden=1024 --n_out=16 >> $output_file

# for setting in {4..7}
# do
#     python main.py --dataset='ba' --setting=$setting --model=MLP --thre=0.2 --n_emb=32 --n_hidden=1024 --n_out=16 >> $output_file

# done


############################################################################################################
# # # Constants
tw="2e7"

# # Setting 1 (specific case without threshold)
python main.py --dataset='msg' --setting=1 --model=MLP --n_emb=32 --n_hidden=64 --n_out=16 >> $output_file

# Settings 2 (specific thresholds)
python main.py --dataset='msg' --setting=2 --model=MLP --thre=5 --tw=$tw --n_emb=32 --n_hidden=64 --n_out=16 >> $output_file
python main.py --dataset='msg' --setting=3 --model=MLP --thre=5 --tw=$tw --n_emb=32 --n_hidden=64 --n_out=16 >> $output_file

for setting in {4..7}
do
    python main.py --dataset='msg' --setting=$setting --model=MLP --thre=0.2 --tw=$tw --n_emb=32 --n_hidden=64 --n_out=16 >> $output_file
done

# tw="2e7"

# # Setting 1 (specific case without threshold)
# python main.py --dataset='msg' --setting=1 --model=MLP2 --n_emb=32 --n_hidden=64 --n_out=16 >> $output_file

# # Settings 2 (specific thresholds)
# python main.py --dataset='msg' --setting=2 --model=MLP2 --thre=5 --tw=$tw --n_emb=32 --n_hidden=64 --n_out=16 >> $output_file
# python main.py --dataset='msg' --setting=3 --model=MLP2 --thre=5 --tw=$tw --n_emb=32 --n_hidden=64 --n_out=16 >> $output_file

# for setting in {4..7}
# do
#     python main.py --dataset='msg' --setting=$setting --model=MLP2 --thre=0.2 --tw=$tw --n_emb=32 --n_hidden=64 --n_out=16 >> $output_file
# done

# python main.py --dataset='tw' --setting=1 --model=MLP --n_emb=128 --n_hidden=1024 --n_out=32 --outfile=1 >> $output_file
# python main.py --dataset='tw' --setting=2 --model=MLP --thre=1 --n_emb=128 --n_hidden=1024 --n_out=32 --outfile=2 >> $output_file
# python main.py --dataset='tw' --setting=3 --model=MLP --thre=1 --n_emb=128 --n_hidden=1024  --n_out=32 --outfile=3 >> $output_file
# python main.py --dataset='tw' --setting=4 --model=MLP --thre=0.2 --n_emb=128 --n_hidden=1024  --n_out=32 --outfile=4 >> $output_file
# python main.py --dataset='tw' --setting=5 --model=MLP --thre=0.2 --n_emb=128 --n_hidden=1024  --n_out=32 --outfile=5 >> $output_file
# python main.py --dataset='tw' --setting=6 --model=MLP --thre=0.05 --n_emb=128 --n_hidden=1024  --n_out=32 --outfile=6 >> $output_file
# python main.py --dataset='tw' --setting=7 --model=MLP --thre=0.05 --n_emb=128 --n_hidden=1024  --n_out=32 --outfile=7 >> $output_file

# python main.py --dataset='tw' --setting=1 --model=MLP2 --n_emb=512 --n_hidden=64 --n_out=16 --outfile=1 >> $output_file
# python main.py --dataset='tw' --setting=2 --model=MLP2 --thre=1 --n_emb=512 --n_hidden=64 --n_out=16 --outfile=2 >> $output_file
# python main.py --dataset='tw' --setting=3 --model=MLP2 --thre=1 --n_emb=512 --n_hidden=64 --n_out=16 --outfile=3 >> $output_file
# python main.py --dataset='tw' --setting=4 --model=MLP2 --thre=0.2 --n_emb=512 --n_hidden=64 --n_out=16 --outfile=4 >> $output_file
# python main.py --dataset='tw' --setting=5 --model=MLP2 --thre=0.2 --n_emb=512 --n_hidden=64 --n_out=16 --outfile=5 >> $output_file
# python main.py --dataset='tw' --setting=6 --model=MLP2 --thre=0.05 --n_emb=512 --n_hidden=64 --n_out=16 --outfile=6 >> $output_file
# python main.py --dataset='tw' --setting=7 --model=MLP2 --thre=0.05 --n_emb=512 --n_hidden=64 --n_out=16 --outfile=7 >> $output_file