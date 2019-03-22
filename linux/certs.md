Given a CA certificate file 'foo.crt', follow these steps to install it on Ubuntu:
First, copy your CA to dir /usr/local/share/ca-certificates/
sudo cp foo.crt /usr/local/share/ca-certificates/foo.crt
then, update CA store
sudo update-ca-certificates
That's all. You should get this output:
Updating certificates in /etc/ssl/certs... 1 added, 0 removed; done.
Running hooks in /etc/ca-certificates/update.d....
Adding debian:foo.pem
done.
done.
No file is needed to edit. Link to your CA is created automatically.
Please note that the certificate filenames have to end in .crt, otherwise the update-ca-certificates script won't pick up on them.

  	 


Use this command to convert *.pem to *.crt: openssl x509 -outform der -in in_file.pem -out out_file.crt – Nelson G. Jun 1 at 10:20

