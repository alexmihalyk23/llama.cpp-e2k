for a in {1..8};do printf "%s;" $a;./main -t $a -m ./models/koala-7B.ggml.q4_0.bin -s 42 -p "Random joke:" -n 32  2>&1 |grep "llama_print_timings:        eval time" | cut -d "(" -f 2 | grep -o -e "[0-9\.]*" ;done
