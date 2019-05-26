FILE=/media/biometric/Data2/FM/Hackathon/output.nii
FILE2=/media/biometric/Data2/FM/Hackathon/mask.png
if [ -f "$FILE" -a "$FILE2" ] 
then
    echo "exists"
else
  	echo "not-exists"
fi


