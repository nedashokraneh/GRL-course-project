# IMDB-M # 

# time python finetune_imdb.py --input_model_file model_gin/supervised.pth \
#                              --dataset imdb-m \
#                              --split random --evaluation f1 \
#                              | tee -a results/imdbm_ft_supervised_f1.txt

# graph infomax # 
echo 'Pre-trained model: infomax | downstream task: IMDB-M'
time python finetune_imdb.py --input_model_file model_gin/infomax.pth \
                             --dataset imdb-m \
                             --split random --evaluation f1 \
                             | tee -a results/imdbm_ft_infomax_f1.txt

# supervised infomax # 
echo 'Pre-trained model: infomax | downstream task: IMDB-M'
time python finetune_imdb.py --input_model_file model_gin/supervised_infomax.pth \
                             --dataset imdb-m \
                             --split random --evaluation f1 \
                             | tee -a results/imdbm_ft_supervised_infomax_f1.txt
# supervised edgepred # 
echo 'Pre-trained model: supervised_edgepred | downstream task: IMDB-M'
time python finetune_imdb.py --input_model_file model_gin/supervised_edgepred.pth \
                             --dataset imdb-m \
                             --split random --evaluation f1 \
                             | tee -a results/imdbm_ft_supervised_edgepred_f1.txt
# supervised masking # 
echo 'Pre-trained model: supervised_masking | downstream task: IMDB-M'
time python finetune_imdb.py --input_model_file model_gin/supervised_masking.pth \
                             --dataset imdb-m \
                             --split random --evaluation f1 \
                             | tee -a results/imdbm_ft_supervised_masking_f1.txt

# supervised contextpred # 
echo 'Pre-trained model: supervised_contextpred | downstream task: IMDB-M'
time python finetune_imdb.py --input_model_file model_gin/supervised_contextpred.pth \
                             --dataset imdb-m \
                             --split random --evaluation f1 \
                             | tee -a results/imdbm_ft_supervised_contextpred_f1.txt



# IMDB-B # 

# graph infomax # 
echo 'Pre-trained model: infomax | downstream task: IMDB-B'
time python finetune_imdb.py --input_model_file model_gin/infomax.pth \
                             --dataset imdb-b \
                             --split random --evaluation f1 \
                             | tee -a results/imdbb_ft_infomax_f1.txt

# supervised # 
echo 'Pre-trained model: supervised | downstream task: IMDB-B'
time python finetune_imdb.py --input_model_file model_gin/supervised.pth \
                             --dataset imdb-b \
                             --split random --evaluation f1 \
                             | tee -a results/imdbb_ft_supervised_f1.txt

# supervised infomax # 
echo 'Pre-trained model: infomax | downstream task: IMDB-B'
time python finetune_imdb.py --input_model_file model_gin/supervised_infomax.pth \
                             --dataset imdb-b \
                             --split random --evaluation f1 \
                             | tee -a results/imdbb_ft_supervised_infomax_f1.txt

# supervised edgepred # 
echo 'Pre-trained model: supervised_edgepred | downstream task: IMDB-B'
time python finetune_imdb.py --input_model_file model_gin/supervised_edgepred.pth \
                             --dataset imdb-b \
                             --split random --evaluation f1 \
                             | tee -a results/imdbb_ft_supervised_edgepred_f1.txt
# supervised masking # 
echo 'Pre-trained model: supervised_masking | downstream task: IMDB-B'
time python finetune_imdb.py --input_model_file model_gin/supervised_masking.pth \
                             --dataset imdb-b \
                             --split random --evaluation f1 \
                             | tee -a results/imdbb_ft_supervised_masking_f1.txt

# supervised contextpred # 
echo 'Pre-trained model: supervised_contextpred | downstream task: IMDB-B'
time python finetune_imdb.py --input_model_file model_gin/supervised_contextpred.pth \
                             --dataset imdb-b \
                             --split random --evaluation f1 \
                             | tee -a results/imdbb_ft_supervised_contextpred_f1.txt
