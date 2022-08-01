rm -rf __pycache__/
rm .DS_Store
rm *.sdimacs*
rm */*.sdimacs*
rm data/raw/repaired* data/raw/reduced*
rm temp.*
rm qelim.res
find . -type d -name  "__pycache__" -exec rm -r {} +
rm -r build
rm -r dist
rm -r *egg*
rm *.pyc
rm justicia/*.pyc

