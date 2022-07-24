rm -f eeVR.zip
git archive HEAD --output eeVR.tar
mkdir eeVR
tar -xvf eeVR.tar -C eeVR
zip -FSr eeVR.zip eeVR
rm -vr eeVR
rm -f eeVR.tar
mkdir eeVR
