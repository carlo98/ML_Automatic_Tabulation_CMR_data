for i in 2 3 4 5 6 7 8 9 10
do
  for j in 6 7 8 9 10
  do
    echo "language ESSENCE' 1.0" >> molnars_molnars-$i-$j.param
    echo "letting k=$i" >> molnars_molnars-$i-$j.param
    echo "letting dommax=$j" >> molnars_molnars-$i-$j.param
  done
done
