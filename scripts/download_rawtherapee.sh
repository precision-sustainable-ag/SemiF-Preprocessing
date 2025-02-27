# run as `source download_rawtherapee.sh` instead of `./download_rawtherapee.sh`

wget https://rawtherapee.com/shared/builds/linux/RawTherapee_5.11_release.AppImage
chmod +x RawTherapee_5.11_release.AppImage
./RawTherapee_5.11_release.AppImage --appimage-extract

# Add conditional check to prevent duplicate entries
#if [[ ":$PATH:" != *":$(pwd)/squashfs-root/usr/bin:"* ]]; then
#    echo 'export PATH="'"$(pwd)"'/squashfs-root/usr/bin/:$PATH"' >> ~/.bashrc
#fi
#echo 'export PATH="'"$(pwd)"'/squashfs-root/usr/bin/:$PATH"' >> ~/.bashrc

export PATH="$(pwd)/squashfs-root/usr/bin/:$PATH"


