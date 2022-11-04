path=./ckpt/
FILES=$path/*
for f in $FILES;do
    echo "==========================" ${f##*/}
    python -u test.py $path${f##*/} ${f##*/}
done

