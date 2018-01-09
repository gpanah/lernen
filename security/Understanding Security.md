## Terms & Relationships

- _AD (Active Directory)_: A comprehensive listing of objects, which can consist of users, systems, resources, and services.  Each type of object has its own schema, which defines the attributes about the type of object. AD is a Microsoft product that runs on Windows server.

- _LDAP (Lightweight Directory Access Protocol)_: A _protocol_ for querying items in a directory service, one example of which is AD.  Other examples of directory services for Linux include FreeIPA, Samba, and OpenSSO.

- _NTLM (**NT** (Networking Technology) **L**AN (**L**ocal **A**rea **N**etwork) **M**anger)_: A suite of Microsoft security protocols.

- _Kerberos_: A network authentication protocol developed by MIT in the late 1980's.  A free implementation is available from MIT, but several commercial offerings are available as well. Microsoft  Active Directory is effectively a Kerberos implementation, as is FreeIPA for Linux (though both provide additional services as well).  Kerberos replaced NTLM as the default authentication mechanism for Windows environments.  
Kerberos distributions have the following key components:
  - _Kerberos Realm_: The set of nodes that share the same Key Distribution Center.
  - _Key Distribution Center_
  - _Authentication Server_
  - _Ticket Granting Service_


  - _SASL (Simple Authentication and Security Layer)_ - a framework for authentication and data security over the internet.  There are several mechanisms, one of which is _GSSAPI (Generic Security Services API)_, which is the standardized API for accessing various implementations, one of which is Kerberos in its various flavors.

  - _SPNEGO (Simple & Protected GSS API Negotiation Mechanism)_ - Mechanism used by client / server to determine with GSSAPI implementations are available for authentication.

  - _Keytab File_


## References

[MIT Kerberos Documentation](https://web.mit.edu/kerberos/krb5-1.15/doc/index.html)
