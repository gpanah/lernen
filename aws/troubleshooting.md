## Weird Latency Issue
I ported an application from one AWS account to another.  In the new account, I provisioned the VPC and all resources.  The differences that I noticed between the two were:
- New VPC used 10. addresses, while the old used 172.
- New Subnets were not default, while the old ones were.
- I believe when I created the new ECS Cluster I had only one subnet defined in the VPC.

Both apps were sitting behind an application loader balancer.

What I noticed is that the new UI would load pretty slowly the first time in most cases, but would eventually load ok in Firefox and Chrome.  However, behind our corporate firewall / proxy, the app would only load well in Chrome.  Every other browser (IE, Firefox, Safari) was EXTREMELY slow (75 secs vs 1 sec).  This browser difference baffled me, especially considering the original page, hosted on AWS just like the new, loaded fine.

I rebuilt the new VPC and Cluster to more closely match the original and things improved.  The only theory that seems to make sense is that communication between the alb and the Cluster was messed up because the Cluster was provisioned when only 1 subnet existed in the VPC.  Something about how routing was being handled meant 
