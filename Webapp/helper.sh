#!/bin/bash

/usr/bin/sshpass -p 'netweb123456' ssh biometric@10.8.10.142 python3 /media/biometric/Data2/FM/Hackathon/reference_test.py

temp=`/usr/bin/sshpass -p 'netweb123456' ssh biometric@10.8.10.142  bash /media/biometric/Data2/FM/Hackathon/check_output.sh`

while [ "$temp" == "not-exists" ]
do
temp=`/usr/bin/sshpass -p 'netweb123456' ssh biometric@10.8.10.142  bash /media/biometric/Data2/FM/Hackathon/check_output.sh`
done

/usr/bin/sshpass -p 'netweb123456' scp -v biometric@10.8.10.142:/media/biometric/Data2/FM/Hackathon/output.nii .
