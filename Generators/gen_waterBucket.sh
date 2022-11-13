for i in 5 6 7 8 9 10
do
  let tmp=$i+2
  let pow_tmp=2**$(($i-1))
  echo "language ESSENCE' 1.0" >> waterBucket_waterBucket$i.param
  echo "letting steps be 7" >> waterBucket_waterBucket$i.param
  echo "$ classical water bucket sizes" >> waterBucket_waterBucket$i.param
  echo "letting SIZE_A be $((2 ** $tmp))" >> waterBucket_waterBucket$i.param
  echo "letting SIZE_B be $((5 * $pow_tmp))" >> waterBucket_waterBucket$i.param
  echo "letting SIZE_C be $((3 * $pow_tmp))" >> waterBucket_waterBucket$i.param
done
