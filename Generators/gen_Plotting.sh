for i in {1..3}
do
  w=$(shuf -i 5-10 -n 1)
  n=$(shuf -i $(($w-2))-$(($w-1)) -n 1)
  java PlottingInstanceGenerator $w $w $n 15
  find . -maxdepth 1 -type f -name "Plotting$w\_$w\_$n\_seed15_*" | shuf -n 100 | xargs -I'{}' mv {} new_plotting_B/
  rm Plotting$w\_$w\_$n\_seed15_*
done
