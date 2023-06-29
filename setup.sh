echo "Hello, SkyPilot!"
echo "Installing dependencies..."
pip install jupyter
pip install -r requirements.txt
# brew install apt
# curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -; sudo brew install -y nodejs
# install repo with the data

folder="svd_directions"
if ! git clone https://github.com/BerenMillidge/svd_directions ; then
    echo "Clone failed because the folder ${folder} exists"
fi

%cd svd_directions
!bash setup.sh
%cd ../

echo "Done!"