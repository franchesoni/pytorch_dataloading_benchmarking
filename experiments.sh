echo "Running experiments..."
# echo "----------------------"
# echo "exp0..."
# python main.py exps/exp0 --n_samples=100
# echo "exp1..."
# python main.py exps/exp1 --batch_size=2 --n_samples=100
# echo "exp2..."
# python main.py exps/exp2 --num_workers=8 --batch_size=2 --n_samples=100
# echo "exp3..."
# python main.py exps/exp3 --num_workers=8 --batch_size=2 --prefetch_factor=2 --n_samples=100
# echo "exp4..."
# python main.py exps/exp4 --num_workers=8 --batch_size=2 --prefetch_factor=2 --spawn --n_samples=100

echo "----------------------"
echo "exp5..."
python main.py exps/exp5 --n_samples=1000
# echo "exp6..."
# python main.py exps/exp6 --batch_size=2 --n_samples=1000
# echo "exp7..."
# python main.py exps/exp7 --num_workers=8 --batch_size=2 --n_samples=1000
# echo "exp8..."
# python main.py exps/exp8 --num_workers=8 --batch_size=2 --prefetch_factor=2 --n_samples=1000
# echo "exp9..."
# python main.py exps/exp9 --num_workers=8 --batch_size=2 --prefetch_factor=2 --spawn --n_samples=1000