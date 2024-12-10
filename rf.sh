# Output file
output_file="final_result/rf.txt"

# Clear the previous content of the output file if it exists
>> $output_file

# Setting 1 (no threshold needed)
# python main.py --dataset='ba' --setting=1 --model=rf >> $output_file

# python main.py --dataset='ba' --setting=2 --model=rf --thre=5 >> $output_file
# python main.py --dataset='ba' --setting=3 --model=rf --thre=5 >> $output_file

# for setting in {4..7}
# do
#     python main.py --dataset='ba' --setting=$setting --model=rf --thre=0.2 >> $output_file

# done

# # # Constants
# tw="2e7"

# # Setting 1 (specific case without threshold)
# python main.py --dataset='msg' --setting=1 --tw=$tw --model=rf >> $output_file
# python main.py --dataset='msg' --setting=2 --thre=5 --tw=$tw --model=rf >> $output_file
# python main.py --dataset='msg' --setting=3 --thre=5 --tw=$tw --model=rf >> $output_file

# for setting in {4..7}
# do

#     python main.py --dataset='msg' --setting=$setting --thre=0.2 --tw=$tw --model=rf >> $output_file

# done

python main.py --dataset='tw' --setting=1 --model=rf --n_emb=64 --n_hidden=1024 --n_out=32 >> $output_file
python main.py --dataset='tw' --setting=2 --model=rf --thre=1 --n_emb=64 --n_hidden=1024 --n_out=32 >> $output_file
python main.py --dataset='tw' --setting=3 --model=rf --thre=1 --n_emb=64 --n_hidden=1024  --n_out=32 >> $output_file
python main.py --dataset='tw' --setting=4 --model=rf --thre=0.2 --n_emb=64 --n_hidden=1024  --n_out=32 >> $output_file
python main.py --dataset='tw' --setting=5 --model=rf --thre=0.2 --n_emb=64 --n_hidden=1024  --n_out=32 >> $output_file
python main.py --dataset='tw' --setting=6 --model=rf --thre=0.05 --n_emb=64 --n_hidden=1024  --n_out=32 >> $output_file
python main.py --dataset='tw' --setting=7 --model=rf --thre=0.05 --n_emb=64 --n_hidden=1024  --n_out=32 >> $output_file