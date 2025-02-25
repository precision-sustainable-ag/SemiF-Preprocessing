# run as `source download_rawtherapee.sh` instead of `./download_rawtherapee.sh`

wget https://rawtherapee.com/shared/builds/linux/RawTherapee_5.11_release.AppImage
chmod +x RawTherapee_5.11_release.AppImage
./RawTherapee_5.11_release.AppImage --appimage-extract

export PATH="$(pwd)/squashfs-root/usr/bin/:$PATH"