# activate your python env before running this script

INGP_HOME=${HOME}/.cache/nuwa/ingp && \

sudo apt install -y build-essential git python3-dev python3-pip libopenexr-dev libxi-dev \
                     libglfw3-dev libglew-dev libomp-dev libxinerama-dev libxcursor-dev && \
# sudo apt install -y cuda-toolkit-12-4 && \    # If you do not have cuda, uncomment this line.
git clone --recursive https://github.com/nvlabs/instant-ngp "$INGP_HOME"  && \
cmake "$INGP_HOME" -B "$INGP_HOME/build" -DCMAKE_BUILD_TYPE=RelWithDebInfo  && \
cmake --build "$INGP_HOME/build" --config RelWithDebInfo -j && \
echo "export PYTHONPATH=\$PYTHONPATH:$INGP_HOME/build" >> "$HOME/.bashrc" && \
#echo "export PYTHONPATH=\$PYTHONPATH:$INGP_HOME/build" >> "$HOME/.zshrc" && \
echo "ingp binary path: $INGP_HOME/instant-ngp" && \
source "$HOME/.bashrc"
#source "$HOME/.zshrc"

# test
echo "Testing if python binding is installed successfully..."
python -c "import pyngp"
