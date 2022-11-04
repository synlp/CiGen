path=./results_all/
FILES=$path/*
for f in $FILES;do
    echo "==========================" ${f##*/}
    python -u metrics_all.py $path${f##*/}
done

