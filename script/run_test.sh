CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch \
--master_port 2502 \
--nproc_per_node=4 \
main_retrieval.py \
--do_eval 1 \
--workers 8 \
--n_display 100 \
--epochs 10 \
--lr 1e-4 \
--coef_lr 1e-3 \
--batch_size 128 \
--batch_size_val 128 \
--anno_path MSRVTT \
--video_path MSRVTT/videos \
--datatype msrvtt \
--max_words 32 \
--max_frames 12 \
--video_framerate 1 \
--output_dir experiments/MSRVTT \
--init_model experiments/MSRVTT/{OUTPUT_PATH}/pytorch_model.bin.{epoch}
