### Create Linux Shortcut Commands

1. At your user home, create a file.
2. In this file, I add commands that I use but can't remember. For example:

  ```
  me ~ $ cat myDockerShortcuts
  function bye_docker() {
    docker rm $(docker ps -a -q)
  }
```
3. In the .bash_profile, source this file:

  ```
  source ~/myDockerShortcuts
  ```

4. Refresh the bash_profile or restart the terminal.
5. To clean up all stopped docker images now sipmly enter `bye_docker`

### Resizing a Volume

In etc the home directory is rather small.  If you run out of space there is a way to move unallocated storage to it.

1. Determine how much space you are using:
```bash
root@rxl0049 lib64 $ df -h
Filesystem            Size  Used Avail Use% Mounted on
/dev/mapper/vg00-lvol3
                      976M  376M  550M  41% /
tmpfs                 939M     0  939M   0% /dev/shm
/dev/sda1             194M   46M  139M  25% /boot
/dev/mapper/vg00-lvol4
                      488M  373M   91M  81% /home
/dev/mapper/vg00-lvol5
                      2.9G  384M  2.4G  14% /opt
/dev/mapper/vg00-lvol6
                      976M  158M  768M  18% /tmp
/dev/mapper/vg00-lvol7
                      2.9G  1.9G  856M  69% /usr
/dev/mapper/vg00-lvol8
                      2.9G  601M  2.1G  22% /var
/dev/mapper/vg00-lv_varopt
                      976M  1.3M  924M   1% /var/opt
```  
2. Determine how much unallocated space there is
```bash
root@rxl0049 lib64 $ vgdisplay
  --- Volume group ---
  VG Name               vg00
  System ID
  Format                lvm2
  Metadata Areas        1
  Metadata Sequence No  9
  VG Access             read/write
  VG Status             resizable
  MAX LV                0
  Cur LV                8
  Open LV               8
  Max PV                0
  Cur PV                1
  Act PV                1
  VG Size               99.78 GiB
  PE Size               32.00 MiB
  Total PE              3193
  Alloc PE / Size       458 / 14.31 GiB
  Free  PE / Size       2735 / 85.47 GiB
  VG UUID               l3MYd8-Ygor-Y6fA-n4sL-Gq7y-irWD-iiEJme
```
3. Find the logical volume associated with the directory that is short on space, in our case /home.  From above we can see that home's filesystem path is /dev/mapper/vg00-lvo14.  The last part of the filesystem (vg00-lvo14) helps us find the appropriate logical volume path.
```
root@rxl0049 lib64 $ lvdisplay
--- Logical volume ---
  LV Path                /dev/vg00/lvol4
  LV Name                lvol4
  VG Name                vg00
  LV UUID                XtwFsN-4d9e-Y6Iu-OcM5-8Bky-jRKD-nM8Wpk
  LV Write Access        read/write
  LV Creation host, time localhost.localdomain, 2016-05-11 14:39:23 -0500
  LV Status              available
  # open                 1
  LV Size                512.00 MiB
  Current LE             16
  Segments               1
  Allocation             inherit
  Read ahead sectors     auto
  - currently set to     256
  Block device           253:2
```
4. Extend the logical volume
```
root@rxl0049 lib64 $ lvextend -L+1G /dev/vg00/lvol4
  Size of logical volume vg00/lvol4 changed from 512.00 MiB (16 extents) to 1.50 GiB (48 extents).
  Logical volume lvol4 successfully resized
```
5. Resize the filesystem so it can see the underlying block is bigger - on Centos 7 need to do something different (http://stackoverflow.com/questions/13362910/trying-to-resize2fs-eb-volume-fails) xfs_growfs /dev/vg00/lvol4
```
root@rxl0049 lib64 $ resize2fs /dev/vg00/lvol4
resize2fs 1.41.12 (17-May-2010)
Filesystem at /dev/vg00/lvol4 is mounted on /home; on-line resizing required
old desc_blocks = 1, new_desc_blocks = 1
Performing an on-line resize of /dev/vg00/lvol4 to 393216 (4k) blocks.
The filesystem on /dev/vg00/lvol4 is now 393216 blocks long.
```
6. re-run df -h to show the new size
```
root@rxl0049 lib64 $df - h
Filesystem            Size  Used Avail Use% Mounted on
/dev/mapper/vg00-lvol3
                      976M  376M  550M  41% /
tmpfs                 939M     0  939M   0% /dev/shm
/dev/sda1             194M   46M  139M  25% /boot
/dev/mapper/vg00-lvol4
                      1.5G  373M  1.1G  27% /home
/dev/mapper/vg00-lvol5
                      2.9G  384M  2.4G  14% /opt
/dev/mapper/vg00-lvol6
                      976M  158M  768M  18% /tmp
/dev/mapper/vg00-lvol7
                      2.9G  1.9G  856M  69% /usr
/dev/mapper/vg00-lvol8
                      2.9G  601M  2.1G  22% /var
/dev/mapper/vg00-lv_varopt
                      976M  1.3M  924M   1% /var/opt
```
