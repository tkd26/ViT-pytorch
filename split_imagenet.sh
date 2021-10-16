train_dirary=()
files="/home/yanai-lab/takeda-m/space0/dataset/decathlon-1.0/data/imagenet12/train/*"
for filepath in $files; do
    train_dirary+=("$filepath")
done

val_dirary=()
files="/home/yanai-lab/takeda-m/space0/dataset/decathlon-1.0/data/imagenet12/val/*"
for filepath in $files; do
    val_dirary+=("$filepath")
done

for id in {0..9}; do
    mkdir /home/yanai-lab/takeda-m/space0/dataset/decathlon-1.0/data/imagenet12/space0/split_train/$id
    mkdir /home/yanai-lab/takeda-m/space0/dataset/decathlon-1.0/data/imagenet12/space0/split_val/$id

    for i in {0..99}; do
        num=`expr 100 \* $id`
        num=`expr $num + $i`
        ln -s ${train_dirary[$num]} /home/yanai-lab/takeda-m/space0/dataset/decathlon-1.0/data/imagenet12/space0/split_train/$id
    done

    for i in {0..99}; do
        num=`expr 100 \* $id`
        num=`expr $num + $i`
        ln -s ${val_dirary[$num]} /home/yanai-lab/takeda-m/space0/dataset/decathlon-1.0/data/imagenet12/space0/split_val/$id
    done
done