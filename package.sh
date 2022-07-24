rm -f eeVR.zip
git archive HEAD --output eeVR.tar
mkdir eeVR
tar -xvf eeVR.tar -C eeVR
rm -f eeVR.tar
zip -FSr eeVR.zip eeVR
rm -vr eeVR
