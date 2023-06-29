conda activate llm_xai
sky launch -c mycluster --use-spot llm-xai.yaml
ssh -L 8888:localhost:8888 mycluster
scp -rp * ubuntu@mycluster:~/
# sky exec mycluster 'sudo snap install code'
# sky exec mycluster 'code --install-extension ms-python.python'
# sky exec mycluster code '--install-extension ms-python.jupyter'
# sky exec mycluster code '--install-extension GitHub.copilot'
# sky exec mycluster code '--install-extension ms-vscode-remote.remote-ssh'