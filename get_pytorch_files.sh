echo "Starting D2L Downloading Script"
echo "Downloading Version 1.0.3 of the book scripts."
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
echo "Completed the Download"
echo "Unzipping the files"
unzip d2l-en.zip
echo "Deleting the zip file"
rm d2l-en.zip
echo "Deleting MACOS files"
rm -r "__MACOSX"
echo "Deleting the non-PyTorch files"
rm -r "tensorflow" && rm -r "mxnet" && rm -r "jax"
echo "D2L Downloading Script Completed."