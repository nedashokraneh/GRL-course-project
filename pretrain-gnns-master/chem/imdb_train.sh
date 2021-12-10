# IMDB-M # 

# time python finetune_imdb.py --input_model_file model_gin/supervised.pth \
#                              --dataset imdb-m \
#                              --split random --evaluation f1 \
#                              | tee -a results/imdbm_ft_supervised_f1.txt

# time python train_supervised.py --dataset imdb-m  \
#                                 --evaluation f1 \
#                                 --output_model_file gin_supervised_imdbm_f1 \
#                                 | tee -a results/gin_supervised_imdbm_f1.txt

# IMDB-B # 

time python train_supervised.py --dataset imdb-b  \
                                --evaluation f1 \
                                --output_model_file gin_supervised_imdbb_f1 \
                                | tee -a results/gin_supervised_imdbb_f1.txt