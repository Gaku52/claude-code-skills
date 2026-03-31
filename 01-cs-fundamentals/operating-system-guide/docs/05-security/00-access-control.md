# Access Control

> The foundation of OS security is access control, which strictly manages "who can access what."

## Learning Objectives

- [ ] Explain the differences between DAC, MAC, and RBAC
- [ ] Understand the Unix permission model
- [ ] Apply the principle of least privilege in practice
- [ ] Design and manage ACLs (Access Control Lists)
- [ ] Configure SELinux / AppArmor policies
- [ ] Minimize privileges using Linux Capabilities
- [ ] Understand the concept of ABAC (Attribute-Based Access Control)
- [ ] Conduct access control audits in production environments


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts

---

## 1. Access Control Models

### 1.1 Fundamental Concepts of Access Control

```
The Three Elements of Access Control (AAA):

  1. Authentication:
     Verifying "Who are you?"
     → Passwords, biometrics, certificates, MFA
     → OS login via /etc/passwd + /etc/shadow
     → Unified authentication through PAM (Pluggable Authentication Modules)

  2. Authorization:
     Determining "What are you allowed to do?"
     → Permissions, ACLs, security policies
     → Security checks within the kernel

  3. Accounting / Auditing:
     Recording "What did you do?"
     → auditd, syslog, journald
     → Meeting compliance requirements

Access Control Decision Flow:
  User Request
       ↓
  [Authentication] → Failure → Access Denied
       ↓ Success
  [Authorization Check] → Denied → Access Denied + Log Entry
       ↓ Permitted
  Resource Access + Audit Log Entry

Security Reference Monitor:
  A mechanism within the kernel that enforces access control
  Requirements:
  1. Complete Mediation: Check every access
  2. Tamper Proof: Cannot be modified
  3. Verifiable: Correctness can be verified
  → Implemented by Linux Security Hooks (LSM: Linux Security Modules)
```

### 1.2 Major Access Control Models

```
Three Access Control Models:

  1. DAC (Discretionary Access Control):
     The file owner freely sets access permissions
     → Standard Unix permissions (rwx)
     → Windows NTFS ACLs
     → Flexible, but vulnerable if the owner misconfigures settings

     Characteristics of DAC:
     - Resource owners can "discretionally" set access permissions
     - Owners can delegate permissions to other users
     - Propagation of permissions is difficult to control (copy problem)
     - Vulnerable to Trojan Horse attacks
       → A malicious program runs with the user's permissions
       → Can steal all files the user can read

     DAC Implementation Examples:
     ┌──────────────────────────────────────────────┐
     │ Unix Permissions                             │
     │ → Three levels: owner / group / other        │
     │ → read(4), write(2), execute(1)             │
     │ → Simple but limited in expressiveness       │
     │                                              │
     │ POSIX ACL                                    │
     │ → Set individual permissions for specific    │
     │   users/groups                               │
     │ → Extends Unix permissions                   │
     │                                              │
     │ Windows DACL                                 │
     │ → Fine-grained control via security          │
     │   descriptors                                │
     │ → Collection of ACEs (Access Control Entries)│
     │ → Supports inheritance and deny rules        │
     └──────────────────────────────────────────────┘

  2. MAC (Mandatory Access Control):
     Follows policies defined by the system administrator (cannot be changed even by the owner)
     → SELinux, AppArmor
     → Required in military and government systems
     → Bell-LaPadula model: Lower levels cannot read higher-level information

     Theoretical Foundations of MAC:
     ┌──────────────────────────────────────────────────┐
     │ Bell-LaPadula Model (Confidentiality-focused):   │
     │                                                  │
     │ Security Levels:                                 │
     │   Top Secret > Secret > Confidential > Unclass   │
     │                                                  │
     │ Simple Security Property (ss-property):          │
     │   "No Read Up": Cannot read higher-level data    │
     │   → A Confidential user cannot read Secret data  │
     │                                                  │
     │ *-property (Star Property):                      │
     │   "No Write Down": Cannot write to lower levels  │
     │   → A Secret user cannot write to Confidential   │
     │   → Prevents information leakage                 │
     │                                                  │
     │ Strong Star Property:                            │
     │   Can only read/write at the same level          │
     └──────────────────────────────────────────────────┘

     ┌──────────────────────────────────────────────────┐
     │ Biba Model (Integrity-focused):                  │
     │                                                  │
     │ Simple Integrity Axiom:                          │
     │   "No Read Down": Cannot read lower-level data   │
     │   → Prevents contamination by untrusted data     │
     │                                                  │
     │ *-Integrity Axiom:                               │
     │   "No Write Up": Cannot write to higher levels   │
     │   → Prevents tampering by untrusted subjects     │
     │                                                  │
     │ Bell-LaPadula and Biba are opposites:            │
     │   Confidentiality vs. integrity trade-off        │
     └──────────────────────────────────────────────────┘

     ┌──────────────────────────────────────────────────┐
     │ Clark-Wilson Model (Commercial integrity):       │
     │                                                  │
     │ Well-Formed Transaction:                         │
     │   Data can only be modified by verified programs │
     │   → Suited for commercial systems like banking   │
     │     and inventory management                     │
     │                                                  │
     │ Separation of Duty:                              │
     │   A single user cannot perform all operations    │
     │   → Separation of duties for fraud prevention    │
     └──────────────────────────────────────────────────┘

  3. RBAC (Role-Based Access Control):
     Assign roles to users, grant permissions to roles
     → Roles such as administrator, developer, viewer
     → AWS IAM, Kubernetes RBAC
     → The standard for large organizations

     RBAC Components:
     ┌──────────────────────────────────────────────┐
     │ User                                         │
     │   ↓ Assignment                               │
     │ Role                                         │
     │   ↓ Grant                                    │
     │ Permission                                   │
     │   ↓ Apply                                    │
     │ Object (Resource)                            │
     └──────────────────────────────────────────────┘

     RBAC Hierarchy (RBAC0-RBAC3):
     - RBAC0 (Flat RBAC): Basic role assignment
     - RBAC1 (Hierarchical RBAC): Role inheritance
       → "Admin" inherits "Developer" permissions
     - RBAC2 (Constrained RBAC): With constraints
       → Mutually exclusive roles (SoD: Separation of Duties)
       → One person cannot be both "approver" and "requester"
     - RBAC3: Integration of RBAC1 + RBAC2

  Comparison:
  ┌──────┬──────────────┬────────────┬──────────────┐
  │ Model│ Decision By  │ Flexibility│ Security     │
  ├──────┼──────────────┼────────────┼──────────────┤
  │ DAC  │ File owner   │ High       │ Medium       │
  │ MAC  │ Sys admin    │ Low        │ High         │
  │ RBAC │ Admin (role) │ Medium     │ High         │
  └──────┴──────────────┴────────────┴──────────────┘
```

### 1.3 ABAC (Attribute-Based Access Control)

```
ABAC (Attribute-Based Access Control):
  Determines access based on "attributes" of users, resources, and the environment
  → Emerged as an extension of RBAC
  → Standardized in NIST SP 800-162
  → Enables more flexible and fine-grained control

  Types of Attributes:
  ┌────────────────────────────────────────────────────┐
  │ Subject Attributes:                                │
  │   → Department, title, clearance level, tenure     │
  │                                                    │
  │ Object Attributes:                                 │
  │   → File type, classification level, creation date │
  │                                                    │
  │ Environment Attributes:                            │
  │   → Access time, location (IP), device type        │
  │                                                    │
  │ Action Attributes:                                 │
  │   → Read, write, delete, execute                   │
  └────────────────────────────────────────────────────┘

  Policy Example:
  "Department=Accounting AND Title=Manager or above AND
   Time=Business hours AND Location=Internal network
   → Allow read/write access to financial reports"

  XACML (eXtensible Access Control Markup Language):
  Standard language for describing ABAC policies
  ┌──────────────────────────────────────────────────┐
  │ PEP (Policy Enforcement Point):                  │
  │   Receives access requests and enforces decisions│
  │                                                    │
  │ PDP (Policy Decision Point):                      │
  │   Determines access based on policies             │
  │                                                    │
  │ PAP (Policy Administration Point):                │
  │   Manages and distributes policies                │
  │                                                    │
  │ PIP (Policy Information Point):                   │
  │   Provides attribute information                  │
  └──────────────────────────────────────────────────┘

  ABAC vs RBAC:
  ┌──────────┬──────────────────┬──────────────────┐
  │ Item     │ RBAC             │ ABAC             │
  ├──────────┼──────────────────┼──────────────────┤
  │ Granularity│ Per role       │ Attribute combos │
  │ Flexibility│ Medium         │ High             │
  │ Management│ Relatively easy │ Complex          │
  │ Dynamic  │ Limited          │ Real-time        │
  │ Use Case │ Internal systems │ Cloud, IoT       │
  │ Standard │ NIST SP 800-xx   │ XACML, ALFA     │
  └──────────┴──────────────────┴──────────────────┘
```

### 1.4 ReBAC (Relationship-Based Access Control)

```
ReBAC (Relationship-Based Access Control):
  Determines access based on "relationships" between entities
  → Google Zanzibar (authorization platform for Google Drive, YouTube)
  → OpenFGA, SpiceDB are OSS implementations

  Basic Concepts:
  ┌────────────────────────────────────────────────────┐
  │ Tuple:                                             │
  │   user:alice → relation:viewer → object:doc:123   │
  │   "alice is a viewer of doc:123"                   │
  │                                                    │
  │ Transitive Relations:                              │
  │   group:engineering#member → relation:viewer       │
  │   → object:folder:shared                           │
  │   user:alice → relation:member → group:engineering│
  │   ∴ alice is a viewer of folder:shared            │
  └────────────────────────────────────────────────────┘

  Zanzibar Architecture:
  - Handles massive scale (authorization for all Google services)
  - Low latency (99th percentile < 10ms)
  - Strong consistency (solves the New Enemy Problem)
  - Graph-based relationship traversal

  Practical Applications:
  → SNS follow/follower permissions
  → File sharing inheritance (folder → file)
  → Permissions based on organizational hierarchy (department head → member resources)
```

---

## 2. Unix Permissions

### 2.1 Basic Permissions

```
Displaying Permissions:
  $ ls -la
  -rwxr-xr-- 1 user group 4096 Jan 1 file.txt
  │├─┤├─┤├─┤
  │ │  │  └── other: r-- (read only)
  │ │  └── group: r-x (read + execute)
  │ └── owner: rwx (read + write + execute)
  └── Type: - (file), d (directory), l (link)

  File Type Reference:
  ┌──────┬──────────────────────────────────┐
  │ Char │ Type                             │
  ├──────┼──────────────────────────────────┤
  │ -    │ Regular file                     │
  │ d    │ Directory                        │
  │ l    │ Symbolic link                    │
  │ c    │ Character device                 │
  │ b    │ Block device                     │
  │ p    │ Named pipe (FIFO)                │
  │ s    │ Socket                           │
  └──────┴──────────────────────────────────┘

  Numeric Representation:
  rwx = 4+2+1 = 7
  r-x = 4+0+1 = 5
  r-- = 4+0+0 = 4
  → chmod 754 file.txt

  Permission Meaning (File vs Directory):
  ┌──────┬──────────────────────┬──────────────────────────┐
  │ Perm │ File                 │ Directory                │
  ├──────┼──────────────────────┼──────────────────────────┤
  │ r(4) │ Read file contents   │ List directory contents  │
  │ w(2) │ Modify file contents │ Create/delete/rename     │
  │ x(1) │ Execute the file     │ Access the directory     │
  └──────┴──────────────────────┴──────────────────────────┘

  Important: Without x permission on a directory, files inside cannot be accessed
  → chmod 644 dir/ allows ls but not cd
  → chmod 711 dir/ allows access if you know the filename inside
```

### 2.2 chmod in Detail

```bash
# Symbolic mode
chmod u+x file.txt      # Add execute permission for owner
chmod g-w file.txt      # Remove write permission from group
chmod o=r file.txt      # Set other permissions to read only
chmod a+r file.txt      # Add read permission for everyone
chmod u+s file.txt      # Set SUID
chmod g+s dir/          # Set SGID
chmod +t dir/           # Set Sticky bit

# Numeric mode
chmod 755 file.txt      # rwxr-xr-x
chmod 644 file.txt      # rw-r--r--
chmod 600 file.txt      # rw------- (sensitive files)
chmod 700 dir/          # rwx------ (private directory)
chmod 4755 file.txt     # SUID + rwxr-xr-x
chmod 2755 dir/         # SGID + rwxr-xr-x
chmod 1777 dir/         # Sticky + rwxrwxrwx

# Recursive change
chmod -R 755 dir/       # Everything under directory (use with caution)
# Safer approach:
find dir/ -type d -exec chmod 755 {} \;  # Directories only
find dir/ -type f -exec chmod 644 {} \;  # Files only
```

### 2.3 Special Permissions

```
Special Permissions:

  SUID (4xxx): Runs with the owner's permissions at execution time
  → /usr/bin/passwd has SUID root (allows regular users to update /etc/shadow)
  → Keep to a minimum due to high security risk

  How SUID Works:
  ┌──────────────────────────────────────────────────┐
  │ Normal execution:                                │
  │   user(uid=1000) → exec(program) → uid=1000     │
  │                                                    │
  │ Execution with SUID:                              │
  │   user(uid=1000) → exec(passwd) → euid=0(root)  │
  │   → Program runs with root privileges            │
  │   → Can write to /etc/shadow                     │
  │   → Privileges revert after program exits        │
  └──────────────────────────────────────────────────┘

  SGID (2xxx): Runs with the group's permissions at execution time
  → For files: Executes with group permissions
  → For directories: Newly created files inherit the directory's group

  Practical Use of SGID Directories:
  ┌──────────────────────────────────────────────────┐
  │ # Create a shared directory                      │
  │ $ sudo mkdir /shared/project                     │
  │ $ sudo chgrp developers /shared/project          │
  │ $ sudo chmod 2775 /shared/project                │
  │                                                    │
  │ # When alice creates a file:                      │
  │ $ touch /shared/project/code.py                  │
  │ $ ls -la /shared/project/code.py                 │
  │ -rw-rw-r-- 1 alice developers ...                │
  │                    ↑ Inherits the directory group │
  │ → All team members can edit the file             │
  └──────────────────────────────────────────────────┘

  Sticky (1xxx): Restricts deletion within a directory to the owner only
  → /tmp has the sticky bit set
  → Cannot delete other users' files

  How the Sticky Bit Works:
  ┌──────────────────────────────────────────────────┐
  │ /tmp (chmod 1777):                                │
  │   rwxrwxrwt ← The trailing 't' is the Sticky bit│
  │                                                    │
  │ Without Sticky bit:                               │
  │   w permission on directory → can delete any file│
  │                                                    │
  │ With Sticky bit:                                  │
  │   File deletion/renaming is limited to:           │
  │   1. The file owner                               │
  │   2. The directory owner                          │
  │   3. root                                          │
  └──────────────────────────────────────────────────┘
```

### 2.4 Ownership Management

```bash
# Change owner
chown alice file.txt            # Change owner to alice
chown alice:developers file.txt # Change owner and group
chown :developers file.txt      # Change group only
chown -R alice:developers dir/  # Change recursively

# Change group
chgrp developers file.txt       # Change group

# Default permissions for new files (umask)
umask                  # Display current umask (e.g., 0022)
umask 0077             # Deny all access to others

# umask calculation:
# Default for files: 666 - umask
# Default for directories: 777 - umask
#
# umask = 0022:
# Files: 666 - 022 = 644 (rw-r--r--)
# Directories: 777 - 022 = 755 (rwxr-xr-x)
#
# umask = 0077:
# Files: 666 - 077 = 600 (rw-------)
# Directories: 777 - 077 = 700 (rwx------)
#
# umask 0077 is recommended for secure servers
```

### 2.5 ACL (Access Control List)

```
ACL (Access Control List):
  Fine-grained control beyond standard permissions
  → Set individual permissions for specific users or groups
  → Used when standard owner/group/other is insufficient

  Types of ACL:
  1. Access ACL: Controls access to files/directories
  2. Default ACL: Default ACL for newly created files within a directory

  Identifying files with ACL:
  $ ls -la
  -rw-rwxr--+ 1 user group 4096 Jan 1 file.txt
             ↑ The '+' mark indicates the presence of an ACL
```

```bash
# View ACL
getfacl file.txt
# file: file.txt
# owner: user
# group: group
# user::rw-
# user:alice:rw-          # rw permissions for alice
# user:bob:r--            # r permission for bob
# group::r-x
# group:devteam:rwx       # rwx permissions for devteam group
# mask::rwx               # Maximum effective permissions
# other::r--

# Set ACL
setfacl -m u:alice:rw file.txt          # Grant read/write to alice
setfacl -m u:bob:r file.txt             # Grant read-only to bob
setfacl -m g:devteam:rwx file.txt       # Grant full permissions to devteam
setfacl -m o::--- file.txt              # Deny all access for other

# Remove ACL
setfacl -x u:alice file.txt             # Remove alice's ACL
setfacl -b file.txt                     # Remove all ACLs

# Default ACL (for directories)
setfacl -d -m u:alice:rw /shared/       # Auto-apply to new files
setfacl -d -m g:devteam:rwx /shared/

# Backup and restore ACL
getfacl -R /shared/ > acl_backup.txt    # Backup
setfacl --restore=acl_backup.txt        # Restore

# Understanding mask:
# mask limits the maximum effective permissions for
# ACL user/group entries and the group owner entry
# Example: If mask::r--, even if ACL is set to rwx,
#          effective permissions become r--
```

### 2.6 Permission Design Patterns in Practice

```
Web Server Permission Design:
  ┌──────────────────────────────────────────────────┐
  │ /var/www/html/                                    │
  │ Owner: www-data:www-data                         │
  │ Permissions:                                     │
  │                                                    │
  │ Static files:                                     │
  │   chmod 644 *.html *.css *.js                    │
  │   → Readable by the web server                   │
  │                                                    │
  │ Executable scripts:                               │
  │   chmod 755 *.cgi *.sh                           │
  │   → Executable but not modifiable                │
  │                                                    │
  │ Upload directory:                                 │
  │   chmod 770 uploads/                              │
  │   → Writable by group members only               │
  │                                                    │
  │ Configuration files:                              │
  │   chmod 600 .env *.conf                          │
  │   → Readable/writable by owner only              │
  │                                                    │
  │ Log directory:                                    │
  │   chmod 750 logs/                                 │
  │   → Readable by group members                    │
  └──────────────────────────────────────────────────┘

SSH Permission Settings:
  ┌──────────────────────────────────────────────────┐
  │ ~/.ssh/                 → chmod 700             │
  │ ~/.ssh/authorized_keys  → chmod 600             │
  │ ~/.ssh/id_rsa           → chmod 600 (priv key) │
  │ ~/.ssh/id_rsa.pub       → chmod 644 (pub key)  │
  │ ~/.ssh/config           → chmod 600             │
  │ ~/.ssh/known_hosts      → chmod 644             │
  │                                                    │
  │ SSH refuses connections if permissions are loose: │
  │ "Permissions 0644 for 'id_rsa' are too open."    │
  │ → Private keys must be readable only by owner    │
  └──────────────────────────────────────────────────┘

Database Permission Design:
  ┌──────────────────────────────────────────────────┐
  │ PostgreSQL:                                       │
  │ /var/lib/postgresql/data/  → Owner: postgres     │
  │ Permissions: 700                                  │
  │ → Database files accessible only by postgres user│
  │                                                    │
  │ MySQL:                                            │
  │ /var/lib/mysql/            → Owner: mysql        │
  │ Permissions: 750                                  │
  │ /etc/mysql/my.cnf          → chmod 644           │
  │ → Config file readable, but only root can modify │
  └──────────────────────────────────────────────────┘
```

---

## 3. Linux Security Modules (LSM)

### 3.1 LSM Architecture

```
Linux Security Modules (LSM) Framework:
  A mechanism that provides security hooks for the Linux kernel
  → Places hooks at various operation points in the kernel
  → Security modules make decisions at these hooks

  LSM Operation Flow:
  User Process
       ↓ System Call
  Kernel (VFS, etc.)
       ↓
  DAC Check → Failure → EACCES
       ↓ Success
  LSM Hook → SELinux/AppArmor/SMACK performs checks
       ↓ Permitted
  Actual Resource Access

  Major LSMs:
  ┌──────────────┬──────────────────────────────────┐
  │ Module       │ Characteristics                  │
  ├──────────────┼──────────────────────────────────┤
  │ SELinux      │ Label-based MAC. Most powerful   │
  │ AppArmor     │ Path-based MAC. Easy to configure│
  │ SMACK        │ Simple MAC. For embedded systems │
  │ TOMOYO       │ Path-based. Has learning mode    │
  │ Yama         │ Supplementary (ptrace restrict)  │
  │ LoadPin      │ Kernel module signature verify   │
  │ Lockdown     │ Restricts kernel features        │
  │ BPF          │ BPF program security             │
  │ Landlock     │ User-space sandboxing            │
  └──────────────┴──────────────────────────────────┘

  LSM Stacking (Linux 5.1+):
  Multiple LSMs can be used simultaneously
  → Combination of SELinux + Yama + Lockdown
  → "Minor LSMs" can always be stacked
  → Only one "major LSM" (SELinux or AppArmor)
    * Major LSM stacking is progressing in Linux 6.x
```

### 3.2 SELinux

```
SELinux (Security-Enhanced Linux, developed by NSA):
  MAC implementation. Labels all processes and files

  SELinux Context:
  ┌──────────────────────────────────────────────────┐
  │ user:role:type:level                              │
  │                                                    │
  │ Example: system_u:system_r:httpd_t:s0             │
  │          │        │        │      │               │
  │          │        │        │      └── MLS level   │
  │          │        │        └── Type (most important)│
  │          │        └── Role                        │
  │          └── SELinux user                         │
  └──────────────────────────────────────────────────┘

  Type Enforcement (TE):
  An httpd_t process can only access httpd_sys_content_t files
  → Even if the web server is compromised, it cannot access other files

  TE Rule Format:
  allow source_type target_type : object_class { permissions };
  allow httpd_t httpd_sys_content_t : file { read open getattr };

  Modes:
  - Enforcing: Denies policy violations + logs
  - Permissive: Logs only (for debugging)
  - Disabled: Inactive

  Basic Operations:
  $ getenforce                # Check current mode
  $ sudo setenforce 0         # Temporarily switch to Permissive
  $ sudo setenforce 1         # Switch to Enforcing

  Checking and Changing Contexts:
  $ ls -Z                     # File context
  $ ps -eZ                    # Process context
  $ id -Z                     # Current user context

  $ sudo chcon -t httpd_sys_content_t /var/www/html/index.html
  $ sudo restorecon -Rv /var/www/html/   # Restore to default
```

```bash
# SELinux Troubleshooting

# Check denial logs
sudo ausearch -m AVC -ts recent
# type=AVC msg=audit(1234567890.123:456):
#   avc: denied { read } for pid=1234 comm="httpd"
#   name="config.php" dev="sda1" ino=789
#   scontext=system_u:system_r:httpd_t:s0
#   tcontext=unconfined_u:object_r:user_home_t:s0
#   tclass=file permissive=0

# Generate policy module with audit2allow
sudo ausearch -m AVC -ts recent | audit2allow -M mypolicy
sudo semodule -i mypolicy.pp

# User-friendly analysis with sealert
sudo sealert -a /var/log/audit/audit.log

# Boolean management (fine-tuning policies)
getsebool -a                                  # List all Booleans
getsebool httpd_can_network_connect            # Check individual
sudo setsebool -P httpd_can_network_connect on # Permanently enable

# Commonly used Booleans:
# httpd_can_network_connect      → HTTPD external connections
# httpd_can_sendmail             → HTTPD mail sending
# httpd_enable_homedirs          → HTTPD home directory access
# allow_ftpd_full_access         → FTP full access
# samba_enable_home_dirs         → Samba home directories

# Persistent file context settings
sudo semanage fcontext -a -t httpd_sys_content_t "/web(/.*)?"
sudo restorecon -Rv /web/

# Port management
sudo semanage port -l | grep http              # List allowed ports
sudo semanage port -a -t http_port_t -p tcp 8080  # Add port

# SELinux user mapping
sudo semanage login -l                         # List mappings
sudo semanage login -a -s staff_u alice        # Map user
```

### 3.3 AppArmor

```
AppArmor (Ubuntu default):
  Path-based MAC. Easier to configure than SELinux
  → Restricts process access via profiles
  → Profiles stored in /etc/apparmor.d/

  AppArmor vs SELinux:
  ┌─────────────┬──────────────────┬──────────────────┐
  │ Item        │ SELinux          │ AppArmor         │
  ├─────────────┼──────────────────┼──────────────────┤
  │ Approach    │ Label-based      │ Path-based       │
  │ Learning    │ High             │ Low              │
  │ Flexibility │ Very high        │ Moderate         │
  │ Default on  │ RHEL/CentOS      │ Ubuntu/SUSE      │
  │ File move   │ Label follows    │ Policy changes   │
  │             │                  │ when path changes│
  │ inode dep.  │ Yes              │ No               │
  │ Network     │ Fine-grained     │ Basic control    │
  └─────────────┴──────────────────┴──────────────────┘

  AppArmor Modes:
  - Enforce: Blocks violations + logs
  - Complain: Logs only (for learning)
  - Unconfined: No restrictions
```

```bash
# AppArmor basic operations
sudo aa-status                   # Check current status
sudo aa-enforce /etc/apparmor.d/usr.sbin.nginx   # Enforce mode
sudo aa-complain /etc/apparmor.d/usr.sbin.nginx  # Complain mode
sudo aa-disable /etc/apparmor.d/usr.sbin.nginx   # Disable

# Create a new profile
sudo aa-genprof /usr/bin/myapp   # Interactively generate profile
# → Operate the app to collect logs → Auto-generate rules

# Analyze logs (violations collected in Complain mode)
sudo aa-logprof                  # Suggest rules from logs
```

```
AppArmor Profile Example (/etc/apparmor.d/usr.sbin.nginx):

  #include <tunables/global>

  /usr/sbin/nginx {
    #include <abstractions/base>
    #include <abstractions/nameservice>

    # Execution permissions
    /usr/sbin/nginx mr,

    # Configuration files
    /etc/nginx/** r,
    /etc/ssl/certs/** r,
    /etc/ssl/private/** r,

    # Web content
    /var/www/** r,

    # Logs
    /var/log/nginx/** w,

    # PID file
    /run/nginx.pid rw,

    # Network
    network inet stream,
    network inet6 stream,

    # Child processes
    /usr/sbin/nginx ix,

    # Temporary files
    /var/lib/nginx/tmp/** rw,

    # deny rules (explicit denial)
    deny /etc/shadow r,
    deny /root/** rwx,
  }

  Profile Permission Symbols:
  ┌──────┬──────────────────────────┐
  │ Sym  │ Meaning                  │
  ├──────┼──────────────────────────┤
  │ r    │ Read                     │
  │ w    │ Write                    │
  │ a    │ Append                   │
  │ x    │ Execute                  │
  │ m    │ Memory map               │
  │ k    │ File lock                │
  │ l    │ Hard link creation       │
  │ ix   │ Inherit execute          │
  │ px   │ Profile execute          │
  │ cx   │ Child profile execute    │
  │ ux   │ Unconfined execute       │
  └──────┴──────────────────────────┘
```

### 3.4 seccomp and Landlock

```
seccomp (Secure Computing Mode):
  Restricts the system calls a process can use
  → Default security for Docker containers
  → Also used in Chrome's sandbox

  seccomp Modes:
  1. Strict Mode: Only allows read, write, _exit, sigreturn
  2. Filter Mode (seccomp-bpf): Fine-grained control with BPF filters

  Docker Default seccomp Profile:
  ┌──────────────────────────────────────────────────┐
  │ Blocks about 40 of the 300+ system calls:        │
  │                                                    │
  │ Blocked system call examples:                     │
  │ - clone (CLONE_NEWUSER): User Namespace creation  │
  │ - mount: Filesystem mounting                      │
  │ - reboot: System reboot                           │
  │ - kexec_load: Kernel replacement                  │
  │ - bpf: Loading BPF programs                       │
  │ - unshare: Creating new Namespaces                │
  │ - ptrace: Debugging other processes               │
  │ - swapon/swapoff: Swap management                 │
  │ - init_module: Loading kernel modules             │
  │                                                    │
  │ Applying a custom profile:                        │
  │ docker run --security-opt seccomp=profile.json    │
  └──────────────────────────────────────────────────┘

Landlock (Linux 5.13+):
  Sandboxing from unprivileged processes
  → Can set access restrictions without root privileges
  → Applications restrict themselves

  Landlock Characteristics:
  - Uses LSM from user space
  - Restrictions are inherited by descendant processes
  - Specialized for filesystem access restriction (v1)
  - Network restrictions also supported (v4, Linux 6.7+)
```

```c
/* Landlock Usage Example (C) */
#include <linux/landlock.h>
#include <sys/prctl.h>
#include <sys/syscall.h>

/* Create a Landlock ruleset */
struct landlock_ruleset_attr ruleset_attr = {
    .handled_access_fs =
        LANDLOCK_ACCESS_FS_READ_FILE |
        LANDLOCK_ACCESS_FS_WRITE_FILE |
        LANDLOCK_ACCESS_FS_EXECUTE,
};

int ruleset_fd = syscall(SYS_landlock_create_ruleset,
    &ruleset_attr, sizeof(ruleset_attr), 0);

/* Rule to allow read/write to /tmp */
struct landlock_path_beneath_attr path_beneath = {
    .allowed_access =
        LANDLOCK_ACCESS_FS_READ_FILE |
        LANDLOCK_ACCESS_FS_WRITE_FILE,
    .parent_fd = open("/tmp", O_PATH | O_CLOEXEC),
};

syscall(SYS_landlock_add_rule, ruleset_fd,
    LANDLOCK_RULE_PATH_BENEATH, &path_beneath, 0);

/* Apply the ruleset (restrictions take effect from here) */
prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);
syscall(SYS_landlock_restrict_self, ruleset_fd, 0);

/* From this point, file access outside /tmp is denied */
```

---

## 4. Principle of Least Privilege

### 4.1 Basic Concept

```
Principle of Least Privilege:
  Grant only the minimum permissions necessary to processes and users

  Theoretical Background:
  One of the 8 design principles proposed by Saltzer & Schroeder (1975)
  in "The Protection of Information in Computer Systems":

  1. Economy of Mechanism: Keep mechanisms simple
  2. Fail-safe Defaults: Default to access denied
  3. Complete Mediation: Check every access
  4. Open Design: Do not rely on design secrecy
  5. Separation of Privilege: Separate privileges
  6. Least Privilege: Minimum privileges           ← This one
  7. Least Common Mechanism: Minimize shared mechanisms
  8. Psychological Acceptability: Ease of use

  Practical Examples:
  ┌──────────────────────────────────────────┐
  │ BAD: Run web server as root             │
  │ GOOD: Run as a dedicated user (www-data)│
  │                                          │
  │ BAD: chmod 777 for full access          │
  │ GOOD: Set only required permissions     │
  │                                          │
  │ BAD: Grant root to an application       │
  │ GOOD: Use capabilities for needed perms │
  │                                          │
  │ BAD: Give admin access to everyone      │
  │ GOOD: Role-based, minimum permissions   │
  │                                          │
  │ BAD: sudo ALL=(ALL) NOPASSWD: ALL       │
  │ GOOD: Allow only specific cmds via sudo │
  └──────────────────────────────────────────┘
```

### 4.2 Linux Capabilities

```
Linux Capabilities:
  Subdivide root privileges and grant only what is needed
  → Split the "all or nothing" root privilege into ~40 fine-grained capabilities

  Major Capabilities:
  ┌──────────────────────────┬────────────────────────────────┐
  │ Capability               │ Description                    │
  ├──────────────────────────┼────────────────────────────────┤
  │ CAP_NET_BIND_SERVICE     │ Bind to ports below 1024       │
  │ CAP_NET_RAW              │ Use RAW sockets                │
  │ CAP_NET_ADMIN            │ Modify network configuration   │
  │ CAP_SYS_PTRACE           │ Debug other processes          │
  │ CAP_SYS_ADMIN            │ Many system admin operations   │
  │ CAP_DAC_OVERRIDE         │ Bypass file permissions        │
  │ CAP_DAC_READ_SEARCH      │ Read and directory search      │
  │ CAP_CHOWN                │ Change file ownership          │
  │ CAP_FOWNER               │ Bypass owner checks            │
  │ CAP_KILL                 │ Signal other users' processes  │
  │ CAP_SETUID               │ Change UID                     │
  │ CAP_SETGID               │ Change GID                     │
  │ CAP_SYS_CHROOT           │ Use chroot                     │
  │ CAP_SYS_TIME             │ Change system clock            │
  │ CAP_AUDIT_WRITE          │ Write to audit log             │
  │ CAP_SYS_RESOURCE         │ Modify resource limits         │
  │ CAP_IPC_LOCK             │ Lock memory (mlock)            │
  │ CAP_SYS_RAWIO            │ Perform raw I/O port operations│
  └──────────────────────────┴────────────────────────────────┘

  Capability Sets:
  ┌──────────────────────────────────────────────────────┐
  │ Permitted:   Maximum capabilities a process can hold │
  │ Effective:   Currently active capabilities           │
  │ Inheritable: Capabilities inherited across execve    │
  │ Bounding:    Upper limit set (cannot exceed)         │
  │ Ambient:     Capabilities passed to unprivileged     │
  │              programs                                │
  └──────────────────────────────────────────────────────┘
```

```bash
# Checking and setting Capabilities

# Check file Capabilities
getcap /usr/bin/ping
# /usr/bin/ping cap_net_raw=ep

# Set Capabilities
sudo setcap cap_net_bind_service=+ep ./server
# → Can bind to port 80 without root

# Set multiple Capabilities
sudo setcap 'cap_net_bind_service,cap_net_raw=+ep' ./server

# Remove Capabilities
sudo setcap -r ./server

# Check process Capabilities
cat /proc/self/status | grep Cap
# CapInh: 0000000000000000  (Inheritable)
# CapPrm: 0000000000000000  (Permitted)
# CapEff: 0000000000000000  (Effective)
# CapBnd: 000001ffffffffff  (Bounding)
# CapAmb: 0000000000000000  (Ambient)

# Decode hex value
capsh --decode=000001ffffffffff

# Launch process with specific Capabilities
sudo capsh --caps="cap_net_bind_service+eip cap_setpcap,cap_setuid,cap_setgid+ep" \
  --keep=1 --user=www-data --addamb=cap_net_bind_service -- -c ./server

# Capabilities management in Docker
docker run --cap-drop ALL --cap-add NET_BIND_SERVICE nginx
# → Drop all Capabilities then add only what is needed
```

### 4.3 Detailed sudo Configuration

```
sudo Configuration (/etc/sudoers):
  Edit with visudo command (includes syntax checking)

  Basic Format:
  user host=(run_as_user) command

  Examples:
  # alice can run all commands on all hosts
  alice ALL=(ALL) ALL

  # alice can restart nginx without a password
  alice ALL=(root) NOPASSWD: /usr/bin/systemctl restart nginx

  # developers group can only run specific commands
  %developers ALL=(root) /usr/bin/docker, /usr/bin/docker-compose

  # bob can run specific commands as a specific user
  bob ALL=(www-data) /usr/bin/php, /usr/bin/composer

  Secure sudoers Design:
  ┌──────────────────────────────────────────────────┐
  │ 1. Minimize NOPASSWD:                            │
  │    → Only for automation scripts                 │
  │    → Always require password for interactive use │
  │                                                    │
  │ 2. Specify commands with absolute paths:          │
  │    → /usr/bin/systemctl (correct)                │
  │    → systemctl (wrong: PATH manipulation attack) │
  │                                                    │
  │ 3. Avoid wildcards:                               │
  │    → ALL=(ALL) /usr/bin/* is dangerous           │
  │    → List required commands individually          │
  │                                                    │
  │ 4. Manage with aliases:                           │
  │    Cmnd_Alias WEBADMIN = /usr/bin/systemctl      │
  │      restart nginx, /usr/bin/certbot              │
  │    %webteam ALL=(root) WEBADMIN                   │
  │                                                    │
  │ 5. Audit sudo logs:                               │
  │    Defaults logfile="/var/log/sudo.log"            │
  │    Defaults log_input, log_output                  │
  │    → Record all sessions                          │
  └──────────────────────────────────────────────────┘

  Dangerous sudo Patterns:
  ┌──────────────────────────────────────────────────┐
  │ Dangerous: alice ALL=(ALL) /usr/bin/vi           │
  │ → Can get root shell via :!/bin/bash in vi      │
  │                                                    │
  │ Dangerous: alice ALL=(ALL) /usr/bin/less         │
  │ → Can get root shell via !bash in less          │
  │                                                    │
  │ Dangerous: alice ALL=(ALL) /usr/bin/find         │
  │ → Can get root shell via find -exec /bin/bash   │
  │                                                    │
  │ Countermeasure: Use sudoedit, or use restricted  │
  │ commands (rnano, other restricted editors)        │
  └──────────────────────────────────────────────────┘
```

### 4.4 PAM (Pluggable Authentication Modules)

```
PAM (Pluggable Authentication Modules):
  Modularizes authentication mechanisms for flexible combination

  PAM Configuration Files:
  Per-service configuration in the /etc/pam.d/ directory
  → /etc/pam.d/sshd, /etc/pam.d/login, /etc/pam.d/sudo

  Configuration Format:
  type  control  module  [options]

  Types:
  ┌──────────┬──────────────────────────────────────┐
  │ auth     │ User authentication (password, MFA)  │
  │ account  │ Account validation (expiry, time)    │
  │ password │ Password change handling             │
  │ session  │ Session management (logs, env setup) │
  └──────────┴──────────────────────────────────────┘

  Controls:
  ┌───────────┬──────────────────────────────────────┐
  │ required  │ Continues to next module even on fail│
  │ requisite │ Immediately denies on failure        │
  │ sufficient│ Skips remaining modules on success   │
  │ optional  │ Does not affect other results        │
  │ include   │ Includes another config file         │
  └───────────┴──────────────────────────────────────┘

  Practical PAM Configuration Example:

  /etc/pam.d/sshd (Hardening SSH Authentication):
  ┌──────────────────────────────────────────────────┐
  │ # Add TOTP (Google Authenticator)                │
  │ auth required pam_google_authenticator.so        │
  │                                                    │
  │ # Password quality requirements                  │
  │ password requisite pam_pwquality.so retry=3      │
  │   minlen=12 dcredit=-1 ucredit=-1 lcredit=-1    │
  │   ocredit=-1                                     │
  │                                                    │
  │ # Login failure lockout                           │
  │ auth required pam_faillock.so                    │
  │   preauth deny=5 unlock_time=900                 │
  │                                                    │
  │ # Access time restriction                         │
  │ account required pam_time.so                     │
  │   → Configure time windows in                    │
  │     /etc/security/time.conf                      │
  │                                                    │
  │ # Login session resource limits                   │
  │ session required pam_limits.so                   │
  │   → Resource limits in /etc/security/limits.conf │
  └──────────────────────────────────────────────────┘
```

---

## 5. Windows Access Control

### 5.1 Windows Security Model

```
Windows Access Control System:

  Security Principals:
  ┌──────────────────────────────────────────────────┐
  │ - User accounts                                   │
  │ - Groups                                          │
  │ - Computer accounts                               │
  │ - Service accounts                                │
  │                                                    │
  │ Each principal is uniquely identified by a        │
  │ SID (Security Identifier)                         │
  │ Example: S-1-5-21-3623811015-3361044348-30300820-1013│
  │          ↑ ↑ ↑   ↑ Domain-specific ID       ↑ RID│
  └──────────────────────────────────────────────────┘

  Access Token:
  Created at login and attached to processes
  ┌──────────────────────────────────────────────────┐
  │ Access Token Contents:                            │
  │ - User SID                                        │
  │ - List of group SIDs                              │
  │ - List of privileges                              │
  │ - Integrity Level                                 │
  │ - Session ID                                      │
  └──────────────────────────────────────────────────┘

  Security Descriptor:
  Attached to each object (files, registry, etc.)
  ┌──────────────────────────────────────────────────┐
  │ - Owner SID: The owner                           │
  │ - Group SID: Primary group                       │
  │ - DACL: Discretionary Access Control List        │
  │   → List of ACEs (Allow/Deny + Perms + SID)     │
  │ - SACL: System Access Control List               │
  │   → Audit settings (log success/failure)         │
  └──────────────────────────────────────────────────┘

  Windows Integrity Levels (MIC: Mandatory Integrity Control):
  ┌──────────────────────────────────────────────────┐
  │ System: Services, kernel objects                  │
  │ High: Admin processes (after UAC elevation)       │
  │ Medium: Normal user processes                     │
  │ Low: Browser sandbox (Internet Explorer)          │
  │ Untrusted: Lowest level                           │
  │                                                    │
  │ → Processes with lower integrity levels cannot    │
  │   write to objects at higher levels (No Write Up) │
  └──────────────────────────────────────────────────┘
```

```powershell
# Windows Access Control Operations (PowerShell)

# View file ACL
Get-Acl C:\Data\report.xlsx | Format-List

# Detailed ACL display
(Get-Acl C:\Data\report.xlsx).Access | Format-Table -AutoSize

# Set ACL
$acl = Get-Acl C:\Data\report.xlsx
$rule = New-Object System.Security.AccessControl.FileSystemAccessRule(
    "DOMAIN\alice", "Read", "Allow")
$acl.SetAccessRule($rule)
Set-Acl C:\Data\report.xlsx $acl

# Disable inheritance
$acl.SetAccessRuleProtection($true, $false)  # Break inheritance, remove existing ACEs
Set-Acl C:\Data\report.xlsx $acl

# icacls command (command line)
icacls C:\Data\report.xlsx
icacls C:\Data\report.xlsx /grant "alice:(R)"
icacls C:\Data\report.xlsx /deny "guest:(W)"
icacls C:\Data\report.xlsx /remove "bob"
icacls C:\Data /grant "developers:(OI)(CI)F" /T  # Apply recursively
```

---

## 6. Cloud Access Control

### 6.1 AWS IAM

```
AWS IAM (Identity and Access Management):
  The core service for controlling access to AWS resources

  Key IAM Components:
  ┌──────────────────────────────────────────────────┐
  │ User: Human users (long-term credentials)        │
  │ Group: Collection of users                        │
  │ Role: Temporary credentials (recommended)         │
  │ Policy: Allow/deny rules for access               │
  │ Identity Provider: Federation with external auth  │
  └──────────────────────────────────────────────────┘

  IAM Policy Structure:
```

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowS3ReadOnly",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::my-bucket",
                "arn:aws:s3:::my-bucket/*"
            ],
            "Condition": {
                "IpAddress": {
                    "aws:SourceIp": "203.0.113.0/24"
                },
                "StringEquals": {
                    "aws:RequestedRegion": "ap-northeast-1"
                }
            }
        },
        {
            "Sid": "DenyDeleteBucket",
            "Effect": "Deny",
            "Action": "s3:DeleteBucket",
            "Resource": "*"
        }
    ]
}
```

```
  IAM Policy Evaluation Logic:
  ┌──────────────────────────────────────────────────┐
  │ 1. Explicit Deny exists → Denied                 │
  │ 2. Not allowed by SCP (Org policy) → Denied     │
  │ 3. Allowed by resource policy → Allowed          │
  │ 4. Allowed by IAM policy → Allowed               │
  │ 5. Not allowed by Permission Boundary → Denied  │
  │ 6. Not allowed by session policy → Denied        │
  │ 7. No allow found → Implicit deny               │
  │                                                    │
  │ Principle: Default deny, explicit allow required  │
  │ Deny Always Wins (Deny takes highest priority)    │
  └──────────────────────────────────────────────────┘

  IAM Best Practices:
  ┌──────────────────────────────────────────────────┐
  │ 1. Do not use the root account for daily tasks   │
  │ 2. Require MFA                                    │
  │ 3. Use IAM Roles (avoid long-term credentials)   │
  │ 4. Design least-privilege policies                │
  │ 5. Detect unused permissions with IAM Access      │
  │    Analyzer                                       │
  │ 6. Set org-wide guardrails with SCPs              │
  │ 7. Leverage tag-based access control (ABAC)       │
  │ 8. Rotate credentials regularly                   │
  │ 9. Audit all API calls with CloudTrail            │
  │ 10. Fine-tune access with condition keys          │
  └──────────────────────────────────────────────────┘
```

### 6.2 Kubernetes RBAC

```yaml
# Kubernetes RBAC Configuration Example

# Role: Defines permissions within a Namespace
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: production
  name: pod-reader
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "watch", "list"]
- apiGroups: [""]
  resources: ["pods/log"]
  verbs: ["get"]

---
# RoleBinding: Binds a Role to users
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: read-pods
  namespace: production
subjects:
- kind: User
  name: alice
  apiGroup: rbac.authorization.k8s.io
- kind: Group
  name: developers
  apiGroup: rbac.authorization.k8s.io
- kind: ServiceAccount
  name: monitoring-sa
  namespace: monitoring
roleRef:
  kind: Role
  name: pod-reader
  apiGroup: rbac.authorization.k8s.io

---
# ClusterRole: Cluster-wide permissions
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: secret-reader
rules:
- apiGroups: [""]
  resources: ["secrets"]
  verbs: ["get", "list"]
  resourceNames: ["app-config", "tls-cert"]  # Specific resources only
```

```
Kubernetes RBAC Best Practices:
┌──────────────────────────────────────────────────┐
│ 1. Prefer Role over ClusterRole                  │
│    → Minimize permissions at Namespace scope      │
│                                                    │
│ 2. Avoid wildcards (*)                            │
│    → verbs: ["*"] is dangerous                    │
│                                                    │
│ 3. Isolate ServiceAccount per Pod                 │
│    → Do not use the default ServiceAccount        │
│                                                    │
│ 4. Audit RBAC                                     │
│    kubectl auth can-i --list --as=alice            │
│    kubectl auth can-i create pods --as=alice       │
│                                                    │
│ 5. Simplify management with Aggregated ClusterRole│
│    → Label-based automatic aggregation            │
│                                                    │
│ 6. Monitor permission usage via audit logs         │
│    → Identify and remove unused permissions        │
└──────────────────────────────────────────────────┘
```

---

## 7. Access Control Auditing and Operations

### 7.1 Auditing Mechanisms

```
Linux auditd (Audit Daemon):
  Monitors and records system calls at the kernel level

  auditd Configuration:
  /etc/audit/auditd.conf    → Daemon configuration
  /etc/audit/rules.d/*.rules → Audit rules

  Audit Rule Examples:
  ┌──────────────────────────────────────────────────┐
  │ # File monitoring (detect changes)               │
  │ -w /etc/passwd -p wa -k user-modify              │
  │ -w /etc/shadow -p wa -k shadow-modify            │
  │ -w /etc/sudoers -p wa -k sudoers-modify          │
  │ -w /etc/ssh/sshd_config -p wa -k sshd-config     │
  │                                                    │
  │ # Monitor file permission changes                │
  │ -a always,exit -F arch=b64 -S chmod,fchmod        │
  │   -F auid>=1000 -F auid!=4294967295 -k perm-change│
  │                                                    │
  │ # Monitor privileged command execution            │
  │ -a always,exit -F path=/usr/bin/sudo -F perm=x    │
  │   -k privileged-cmd                                │
  │                                                    │
  │ # User authentication events                      │
  │ -w /var/log/faillog -p wa -k login-failures       │
  │ -w /var/log/lastlog -p wa -k last-login           │
  └──────────────────────────────────────────────────┘
```

```bash
# Search audit logs
ausearch -k user-modify            # Search by key
ausearch -m USER_AUTH -ts today    # Today's auth events
ausearch -ui 1000 -ts recent       # Recent events for a specific user

# Generate audit reports
aureport --summary                 # Summary
aureport --auth                    # Authentication report
aureport --file --failed           # Failed file access
aureport --login --failed          # Failed logins

# Real-time monitoring
tail -f /var/log/audit/audit.log | ausearch --interpret
```

### 7.2 Security Audit Checklist

```
Periodic Access Control Audit Checklist:

  [ ] Inventory SUID/SGID files
      find / -perm -4000 -o -perm -2000 -type f 2>/dev/null
      → Check for unauthorized SUID files

  [ ] Check world-writable files
      find / -perm -002 -type f ! -path "/proc/*" 2>/dev/null
      → Ensure sensitive data is not writable by everyone

  [ ] Check files with no owner
      find / -nouser -o -nogroup 2>/dev/null
      → Ensure no files remain from deleted users

  [ ] Review sudo configuration
      sudo -l                      # Your sudo permissions
      visudo -c                    # Syntax check sudoers

  [ ] Check for unnecessary user accounts
      awk -F: '$7 !~ /(nologin|false)$/ {print $1}' /etc/passwd
      → List of accounts that can log in

  [ ] Review password policy
      chage -l username            # Password expiration
      grep -E '^PASS' /etc/login.defs  # Password policy

  [ ] Review SSH configuration
      /etc/ssh/sshd_config:
      - PermitRootLogin no
      - PasswordAuthentication no
      - PubkeyAuthentication yes
      - MaxAuthTries 3
      - AllowUsers / AllowGroups settings

  [ ] Review firewall rules
      iptables -L -n -v            # iptables
      nft list ruleset             # nftables
      ufw status verbose           # UFW

  [ ] Check SELinux/AppArmor status
      getenforce                   # SELinux
      aa-status                    # AppArmor
      → Ensure Enforcing/Enforce mode is active

  [ ] Review logs
      /var/log/auth.log            # Auth log
      /var/log/secure              # Secure log (RHEL-based)
      journalctl -u sshd --since today  # SSH logs
```

### 7.3 Compliance

```
Major Compliance Standards and Access Control Requirements:

  PCI DSS (Payment Card Industry):
  ┌──────────────────────────────────────────────────┐
  │ Req 7: Access limited to business need-to-know   │
  │ Req 8: Unique identification of users             │
  │ Req 10: Track and monitor all access to network  │
  │         resources and cardholder data              │
  │                                                    │
  │ Specific Measures:                                │
  │ - Implement role-based access control              │
  │ - Prohibit shared accounts                         │
  │ - Regular access rights review (quarterly)         │
  │ - Retain all access logs for one year              │
  └──────────────────────────────────────────────────┘

  SOX (Sarbanes-Oxley Act):
  ┌──────────────────────────────────────────────────┐
  │ Implement Separation of Duties                    │
  │ → Separate approvers and executors                │
  │ → Developers do not directly access production   │
  │                                                    │
  │ Record change management processes                │
  │ → Associate all changes with ticket numbers       │
  │ → Retain evidence of approval workflows           │
  └──────────────────────────────────────────────────┘

  GDPR (EU General Data Protection Regulation):
  ┌──────────────────────────────────────────────────┐
  │ Data Minimization Principle                       │
  │ → Collect and retain only the minimum data needed│
  │ → Set access permissions to minimum as well       │
  │                                                    │
  │ Access Log Retention                              │
  │ → Record access history to personal data          │
  │ → Make disclosable upon data subject request      │
  └──────────────────────────────────────────────────┘

  CIS Benchmarks:
  → Security configuration guidelines for OSes
  → Verify compliance with automated scanning tools (OpenSCAP, Lynis)
  → Two tiers: Level 1 (basic) and Level 2 (hardened)
```

---

## Hands-on Exercises

### Exercise 1: [Beginner] -- Permission Operations

```bash
# Manipulate file permissions
touch test.txt
chmod 644 test.txt && ls -la test.txt
chmod u+x test.txt && ls -la test.txt
chmod o-r test.txt && ls -la test.txt

# Change ownership
sudo chown root:root test.txt

# Check and set umask
umask
umask 0077
touch secret.txt && ls -la secret.txt   # Should be rw-------
umask 0022  # Reset
```

### Exercise 2: [Beginner] -- Setting Up ACLs

```bash
# Set and verify ACLs
touch shared.txt
setfacl -m u:alice:rw shared.txt
setfacl -m u:bob:r shared.txt
setfacl -m g:developers:rw shared.txt
getfacl shared.txt

# Set default ACL
mkdir /tmp/shared_dir
setfacl -d -m g:developers:rw /tmp/shared_dir
touch /tmp/shared_dir/newfile.txt
getfacl /tmp/shared_dir/newfile.txt  # Default ACL should be applied
```

### Exercise 3: [Advanced] -- Security Auditing

```bash
# Search for SUID files (important for security auditing)
find / -perm -4000 -type f 2>/dev/null

# Search for writable files
find /etc -writable -type f 2>/dev/null

# Search for files with no owner
find / -nouser -o -nogroup 2>/dev/null | head -20

# List accounts that can log in
awk -F: '$7 !~ /(nologin|false)$/ {print $1, $7}' /etc/passwd

# Search for accounts with no password (empty)
sudo awk -F: '($2 == "" || $2 == "!") {print $1}' /etc/shadow

# Check recently modified files (breach investigation)
find /etc -mtime -1 -type f 2>/dev/null
find /usr/bin -mtime -1 -type f 2>/dev/null
```

### Exercise 4: [Advanced] -- Using Capabilities

```bash
# Configure privileged port binding
gcc -o webserver webserver.c
sudo setcap cap_net_bind_service=+ep ./webserver
getcap ./webserver
# → Can listen on port 80 without root

# Check current Capabilities
cat /proc/self/status | grep Cap
capsh --print

# Restrict Capabilities in Docker containers
docker run --cap-drop ALL \
  --cap-add NET_BIND_SERVICE \
  --cap-add CHOWN \
  -p 80:80 nginx
```

### Exercise 5: [Production] -- SELinux Troubleshooting

```bash
# Resolve web server access issues on an SELinux-enabled system

# 1. Identify the problem
sudo ausearch -m AVC -ts recent

# 2. Detailed analysis
sudo sealert -a /var/log/audit/audit.log

# 3. Check file contexts
ls -Z /var/www/html/
ls -Z /home/user/public_html/

# 4. Set the correct context
sudo semanage fcontext -a -t httpd_sys_content_t "/web(/.*)?"
sudo restorecon -Rv /web/

# 5. Check and set Booleans
getsebool httpd_enable_homedirs
sudo setsebool -P httpd_enable_homedirs on
```

### Exercise 6: [Production] -- Comprehensive Access Control Design

```bash
# Access control design for a web application server

# 1. Create a dedicated user and group
sudo groupadd webapp
sudo useradd -r -g webapp -s /sbin/nologin webapp-user

# 2. Create directory structure and set permissions
sudo mkdir -p /opt/webapp/{app,config,data,logs,tmp}
sudo chown -R webapp-user:webapp /opt/webapp
sudo chmod 750 /opt/webapp
sudo chmod 750 /opt/webapp/app
sudo chmod 700 /opt/webapp/config    # Owner only for config
sudo chmod 770 /opt/webapp/data      # Group can also write
sudo chmod 750 /opt/webapp/logs      # Group can read
sudo chmod 700 /opt/webapp/tmp       # Owner only for temp files

# 3. Set deploy user permissions
sudo usermod -aG webapp deploy-user
setfacl -R -m u:deploy-user:rwx /opt/webapp/app
setfacl -R -d -m u:deploy-user:rwx /opt/webapp/app

# 4. Set Capabilities (privileged port binding without root)
sudo setcap cap_net_bind_service=+ep /opt/webapp/app/server

# 5. Configure sudo (allow only deploy operations)
# Add the following via visudo:
# deploy-user ALL=(webapp-user) NOPASSWD: /usr/bin/systemctl restart webapp

# 6. Set audit rules
sudo auditctl -w /opt/webapp/config -p wa -k webapp-config
sudo auditctl -w /opt/webapp/app -p wa -k webapp-deploy
```

---


## FAQ

### Q1: What is the most important point when studying this topic?

Gaining practical experience is the most important. Understanding deepens not just through theory, but by actually writing code and observing how it works.

### Q2: What are common mistakes beginners make?

Skipping the basics and jumping to advanced topics. We recommend thoroughly understanding the fundamental concepts explained in this guide before moving on to the next step.

### Q3: How is this used in real-world practice?

Knowledge of this topic is frequently applied in daily development work. It becomes particularly important during code reviews and architecture design.

---

## Summary

| Concept | Key Points |
|---------|-----------|
| DAC | Owner sets permissions. Unix standard. Flexible but vulnerable to Trojan Horse attacks |
| MAC | Enforced by admin. SELinux, AppArmor. Bell-LaPadula, Biba models |
| RBAC | Role-based. AWS IAM, K8s. Standard for large organizations |
| ABAC | Attribute-based. XACML. Most flexible model |
| ReBAC | Relationship-based. Google Zanzibar. SNS, file sharing |
| ACL | Fine-grained control beyond standard permissions. POSIX ACL |
| Capabilities | Subdivision of root privileges. Enables least privilege |
| LSM | Framework for SELinux/AppArmor. Kernel-level MAC |
| PAM | Modular authentication. MFA, lockout, password policies |
| Least Privilege | Grant only minimum necessary permissions. Fundamental security principle |
| Auditing | auditd, CloudTrail. Essential for compliance |

---

## Recommended Next Guides

---

## References
1. Bishop, M. "Computer Security: Art and Science." 2nd Ed, Addison-Wesley, 2018.
2. Saltzer, J. H. & Schroeder, M. D. "The Protection of Information in Computer Systems." Proceedings of the IEEE, 1975.
3. Smalley, S. et al. "Implementing SELinux as a Linux Security Module." NAI Labs Report, 2001.
4. Ferraiolo, D. F. et al. "Proposed NIST Standard for Role-Based Access Control." ACM TISSEC, 2001.
5. Hu, V. C. et al. "Guide to Attribute Based Access Control (ABAC) Definition and Considerations." NIST SP 800-162, 2014.
6. Zanzibar: "Google's Consistent, Global Authorization System." USENIX ATC, 2019.
7. Red Hat. "SELinux User's and Administrator's Guide." Red Hat Enterprise Linux Documentation, 2024.
8. Canonical. "AppArmor Documentation." Ubuntu Security Documentation, 2024.
9. AWS. "IAM Best Practices." AWS Documentation, 2024.
10. NIST. "SP 800-53: Security and Privacy Controls for Information Systems and Organizations." Rev. 5, 2020.
