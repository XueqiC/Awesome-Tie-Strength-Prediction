# Output file
# output_file="final_result/tran.txt"
output_file="final_result/reweight.txt"

# Clear the previous content of the output file if it exists
>> $output_file

# Setting 1 (no threshold needed)
# python main.py --dataset='ba' --setting=1 --model=Tran --n_emb=1024 --n_hidden=32 --n_out=16 >> $output_file


# python main.py --dataset='ba' --setting=2 --model=Tran --thre=5 --n_emb=1024 --n_hidden=32 --n_out=16 >> $output_file
# python main.py --dataset='ba' --setting=3 --model=Tran --thre=5 --n_emb=1024 --n_hidden=32 --n_out=16 >> $output_file

# for setting in {4..7}
# do
#     python main.py --dataset='ba' --setting=$setting --model=Tran --thre=0.2 --n_emb=1024 --n_hidden=32 --n_out=16 >> $output_file
# done



# # # Constants
tw="2e7"

# # Setting 1 (specific case without threshold)
# python main.py --dataset='msg' --setting=1 --model=Tran --tw=$tw --n_emb=64 --n_hidden=256 --n_out=64 >> $output_file


# python main.py --dataset='msg' --setting=2 --model=Tran --thre=5 --tw=$tw  --n_emb=64 --n_hidden=256 --n_out=64 >> $output_file
# python main.py --dataset='msg' --setting=3 --model=Tran --thre=5 --tw=$tw  --n_emb=64 --n_hidden=256 --n_out=64 >> $output_file

# for setting in {4..7}
# do
#     python main.py --dataset='msg' --setting=$setting --model=Tran --thre=0.2 --tw=$tw  --n_emb=64 --n_hidden=256 --n_out=64 >> $output_file

# done


# python main.py --dataset='tw' --setting=1 --model=Tran --n_emb=128 --n_hidden=64 --n_out=128 >> $output_file
# python main.py --dataset='tw' --setting=2 --model=Tran --thre=1 --n_emb=128 --n_hidden=64 --n_out=128 >> $output_file
# python main.py --dataset='tw' --setting=3 --model=Tran --thre=1 --n_emb=128 --n_hidden=64 --n_out=128 >> $output_file
# python main.py --dataset='tw' --setting=4 --model=Tran --thre=0.2 --n_emb=128 --n_hidden=64 --n_out=128 >> $output_file
# python main.py --dataset='tw' --setting=5 --model=Tran --thre=0.2 --n_emb=128 --n_hidden=64 --n_out=128 >> $output_file
# python main.py --dataset='tw' --setting=6 --model=Tran --thre=0.05 --n_emb=128 --n_hidden=64 --n_out=128 >> $output_file
# python main.py --dataset='tw' --setting=7 --model=Tran --thre=0.05 --n_emb=128 --n_hidden=64 --n_out=128 >> $output_file