# make gif from everything in ./out/
# Usage: ./makegif.sh
convert -delay 20 -loop 0 ./out/*.png training.gif