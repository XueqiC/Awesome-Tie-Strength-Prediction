# Output file
# output_file="./final_result/gcn_future.txt"
output_file="./final_result/reweight.txt"

# Clear the previous content of the output file if it exists
>> $output_file


## Definition I
# python main.py --dataset='ba' --setting=1 --model=GCN --n_emb=32 --n_hidden=1024 --n_out=16 >> $output_file

## Definition II and III (global threshold)
# python main.py --dataset='ba' --setting=2 --model=GCN --thre=5 --n_emb=32 --n_hidden=1024 --n_out=16 >> $output_file
# python main.py --dataset='ba' --setting=3 --model=GCN --thre=5 --n_emb=32 --n_hidden=1024 --n_out=16 >> $output_file

## Definition IV to VII (local threshold)
# for setting in {4..7}
# do
#     python main.py --dataset='ba' --setting=$setting --model=GCN --thre=0.2 --n_emb=32 --n_hidden=1024 --n_out=16 >> $output_file

# done

# # # Constants
tw="2e7"

# python main.py --dataset='msg' --setting=1 --model=GCN --n_emb=32 --n_hidden=512 --n_out=32 >> $output_file
# python main.py --dataset='msg' --setting=2 --model=GCN --thre=5 --tw=$tw --n_emb=32 --n_hidden=512 --n_out=32 >> $output_file
# python main.py --dataset='msg' --setting=3 --model=GCN --thre=5 --tw=$tw --n_emb=32 --n_hidden=512 --n_out=32 >> $output_file
# for setting in {4..7}
# do
#     python main.py --dataset='msg' --setting=$setting --model=GCN --thre=0.2 --tw=$tw --n_emb=32 --n_hidden=512 --n_out=32 >> $output_file
# done


# python main.py --dataset='tw' --setting=1 --model=GCN --n_emb=32 --n_hidden=1024 --n_out=16 --outfile=1 >> $output_file
# python main.py --dataset='tw' --setting=2 --model=GCN --thre=1 --n_emb=32 --n_hidden=1024 --n_out=16 --outfile=2  >> $output_file
# python main.py --dataset='tw' --setting=3 --model=GCN --thre=1 --n_emb=32 --n_hidden=1024 --n_out=16 --outfile=3 >> $output_file
# python main.py --dataset='tw' --setting=4 --model=GCN --thre=0.2 --n_emb=32 --n_hidden=1024 --n_out=16 --outfile=4 >> $output_file
# python main.py --dataset='tw' --setting=5 --model=GCN --thre=0.2 --n_emb=32 --n_hidden=1024 --n_out=16 --outfile=5 >> $output_file
# python main.py --dataset='tw' --setting=6 --model=GCN --thre=0.05 --n_emb=32 --n_hidden=1024 --n_out=16 --outfile=6 >> $output_file
# python main.py --dataset='tw' --setting=7 --model=GCN --thre=0.05 --n_emb=32 --n_hidden=1024 --n_out=16 --outfile=7 >> $output_file
