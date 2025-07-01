#!/usr/bin/python
# pyright: reportShadowedImports=false

""" This profile is for running StopSec experiments on POWDER.

This profile creates a temporary Local Area Network (LAN) for the duration of the exepriment. 
The user can select the number of the nodes in the LAN. 
By deffault, one of the nodes will be the database-server and one another will be the Primary-user. 
The primary-user and secondary-user nodes are connected to x310 radios (rooftop base stations) for over-the-air transmissions.
Again, by deffault, the first node you select will be the database-server and the second node will be the primary-user. 
The third node will be the seconday-user1, the fourth will be secondary-user2 etc.

Instructions:

**1) Instantiate this profile with appropriate parameters**

At the "Parameterize" step, add radios that are needed for your planned experiment. Also, speceify the freqeuncy ranges if you are planning to use transmitter(s) in your experiment. 

Once you have these parameters selected, click through the rest of the profile and then click "Finish" to instantiate.  It will take 10 to 15 minutes for the experiment to finish setting up.  Once it is "green", proceed to the next step.

**2) Open SSH sessions**

Use the following commands to start ssh to each of the nodes:
```
ssh <username>@<orch_node_hostname> - Y
```

"""

# Import the Portal object.
import geni.portal as portal

# Import the ProtoGENI library.
import geni.rspec.pg as pg

#import PG interface
import geni.rspec.pg as rspec
# Emulab specific extensions.
import geni.rspec.emulab as emulab

import geni.rspec.emulab.pnext as pn
import geni.rspec.emulab.spectrum as spectrum
import geni.rspec.emulab.route as route
import geni.rspec.igext as ig


COMP_MANAGER_ID = "urn:publicid:IDN+emulab.net+authority+cm"
DEFAULT_NODE_TYPE = "d430"
# GNURADIO_DISK_IMAGE = "urn:publicid:IDN+emulab.net+image+emulab-ops//UBUNTU18-64-GR38-PACK"  # shout still requires this image

# GNURADIO_DISK_IMAGE =  "urn:publicid:IDN+emulab.net+image+emulab-ops//UBUNTU22-64-STD"
GNURADIO_DISK_IMAGE = "urn:publicid:IDN+emulab.net+image+emulab-ops//UBUNTU22-64-GR310"
VANILLA_OS_IMAGES = [
    "urn:publicid:IDN+emulab.net+image+emulab-ops//UBUNTU22-64-GR310",
    "urn:publicid:IDN+emulab.net+image+emulab-ops:UBUNTU22-64-GR310",
    "urn:publicid:IDN+emulab.net+image+emulab-ops//UBUNTU22-64-STD",
    "urn:publicid:IDN+emulab.net+image+emulab-ops//UBUNTU20-64-STD",
    "urn:publicid:IDN+emulab.net+image+emulab-ops//UBUNTU18-64-STD",
]

# Create a portal context, needed to defined parameters
pc = portal.Context()

# List of CBRS rooftop X310 radios.
cbrs_radios = [
    ("cbrssdr1-bes", "Behavioral"),
    ("cbrssdr1-browning", "Browning"),
    ("cbrssdr1-dentistry", "Dentistry"),
    ("cbrssdr1-fm", "Friendship Manor"),
    ("cbrssdr1-hospital", "Hospital"),
    ("cbrssdr1-honors", "Honors"),
    ("cbrssdr1-meb", "MEB"),
    ("cbrssdr1-smt", "SMT"),
    ("cbrssdr1-ustar", "USTAR"),
]
# Set of CBRS X310 radios to allocate
pc.defineStructParameter(
    "cbrs_radio_sites", "Rooftop CBRS Radio Sites", [],
    multiValue=True,
    min=2,
    multiValueTitle="CBRS X310 radios to allocate.",
    members=[
        portal.Parameter(
            "device",
            "CBRS Radio Site",
            portal.ParameterType.STRING,
            cbrs_radios[0], cbrs_radios,
            longDescription="CBRS X310 radio will be allocated from selected site."
        ),
    ]
)

# Frequency ranges to declare for this experiment
pc.defineStructParameter(
    "freq_ranges", "Frequency Ranges To Transmit In", [],
    multiValue=True,
    min=0,
    multiValueTitle="Frequency ranges to be used for transmission.",
    members=[
        portal.Parameter(
            "freq_min",
            "Frequency Range Min",
            portal.ParameterType.BANDWIDTH,
            3350.0,
            longDescription="Values are rounded to the nearest kilohertz."
        ),
        portal.Parameter(
            "freq_max",
            "Frequency Range Max",
            portal.ParameterType.BANDWIDTH,
            3650.0,
            longDescription="Values are rounded to the nearest kilohertz."
        ),
    ]
)

pc.defineParameter(
    name="gnuradio_os_image",
    description="Select OS image",
    defaultValue="",
    typ=portal.ParameterType.STRING,
    longDescription="Select the OS image for the GNU Radio nodes."
)

pc.defineParameter(
    name="other_os_image",
    description="Select OS image",
    typ=portal.ParameterType.STRING,
    defaultValue=VANILLA_OS_IMAGES[2],
    legalValues=VANILLA_OS_IMAGES,
    longDescription="Most clusters have this set of images, " +
                    "pick your favorite one."
)

pc.defineParameter(
    name="gnuradio_node_type",
    description="Optional physical node type for nodes that will transmit/receive RF",
    defaultValue=DEFAULT_NODE_TYPE,
    typ=portal.ParameterType.STRING,
    longDescription="Pick a single physical node type (pc3000,d710,etc) " +
                    "instead of letting the resource mapper choose for you."
)

pc.defineParameter(
    name="other_node_type",
    description="Optional physical node type for nodes that will not transmit/receive RF",
    defaultValue="",
    typ=portal.ParameterType.STRING,
    longDescription="Pick a single physical node type (pc3000,d710,etc) " +
                    "instead of letting the resource mapper choose for you."
)

# Optionally create XEN VMs instead of allocating bare metal nodes.
pc.defineParameter(
    name="useVMs",
    description="Use XEN VMs",
    defaultValue=False,
    typ=portal.ParameterType.BOOLEAN,
    longDescription="Create XEN VMs instead of allocating bare metal nodes."
)

# Optionally start X11 VNC server.
pc.defineParameter(
    name="startVNC",
    description="Start X11 VNC on your nodes",
    defaultValue=True,
    typ=portal.ParameterType.BOOLEAN,
    longDescription="Start X11 VNC server on your nodes. There will be " +
                    "a menu option in the node context menu to start a browser based VNC " +
                    "client. Works really well, give it a try!"
)

# Optional link speed, normally the resource mapper will choose for you based on node availability
pc.defineParameter(
    name="linkSpeed",
    description="Link Speed",
    typ=portal.ParameterType.INTEGER,
    defaultValue=0,
    legalValues=[
        (0,"Any"),
        (100000,"100Mb/s"),
        (1000000,"1Gb/s"),
        (10000000,"10Gb/s"),
        (25000000,"25Gb/s"),
        (100000000,"100Gb/s")
    ],
    advanced=True,
    longDescription="A specific link speed to use for your lan. Normally the resource " +
                    "mapper will choose for you based on node availability and the optional physical type."
)
                   
# For very large lans you might to tell the resource mapper to override the bandwidth constraints
# and treat it a "best-effort"
pc.defineParameter(
    name="bestEffort",
    description="Best Effort",
    defaultValue=False,
    typ=portal.ParameterType.BOOLEAN,
    advanced=True,
    longDescription="For very large lans, you might get an error saying 'not enough bandwidth.' " +
                    "This options tells the resource mapper to ignore bandwidth and assume you know what you " +
                    "are doing, just give me the lan I ask for (if enough nodes are available)."
)


# Sometimes you want all of nodes on the same switch, Note that this option can make it impossible
# for your experiment to map.
pc.defineParameter(
    name="sameSwitch",
    description="No Interswitch Links",
    defaultValue=False,
    typ=portal.ParameterType.BOOLEAN,
    advanced=True,
    longDescription="Sometimes you want all the nodes connected to the same switch. " +
                    "This option will ask the resource mapper to do that, although it might make " +
                    "it imppossible to find a solution. Do not use this unless you are sure you need it!"
)

# Optional ephemeral blockstore
pc.defineParameter(
    name="tempFileSystemSize",
    description="Temporary Filesystem Size",
    defaultValue=0,
    typ=portal.ParameterType.INTEGER,
    advanced=True,
    longDescription="The size in GB of a temporary file system to mount on each of your " +
                    "nodes. Temporary means that they are deleted when your experiment is terminated. " +
                    "The images provided by the system have small root partitions, so use this option " +
                    "if you expect you will need more space to build your software packages or store " +
                    "temporary files."
)
                   
# Instead of a size, ask for all available space. 
pc.defineParameter(
    name="tempFileSystemMax",
    description="Temp Filesystem Max Space",
    defaultValue=False,
    typ=portal.ParameterType.BOOLEAN,
    advanced=True,
    longDescription="Instead of specifying a size for your temporary filesystem, " +
                    "check this box to allocate all available disk space. Leave the size above as zero."
)

pc.defineParameter(
    name="tempFileSystemMount",
    description="Temporary Filesystem Mount Point",
    defaultValue="/mydata",
    typ=portal.ParameterType.STRING,
    advanced=True,
    longDescription="Mount the temporary file system at this mount point; in general you " +
                    "you do not need to change this, but we provide the option just in case your software " +
                    "is finicky."
)

pc.defineParameter(
    name="exclusiveVMs",
    description="Force use of exclusive VMs",
    defaultValue=True,
    typ=portal.ParameterType.BOOLEAN,
    advanced=True,
    longDescription="When True and useVMs is specified, setting this will force allocation " +
                    "of exclusive VMs. When False, VMs may be shared or exclusive depending on the policy " +
                    "of the cluster."
)

pc.defineParameter(
    name="alloc_shuttles",
    description="Allocate all routes (mobile endpoints)",
    defaultValue=False,
    typ=portal.ParameterType.BOOLEAN
)

params = pc.bindParameters()
pc.verifyParameters()
request = pc.makeRequestRSpec()

# Create database-server node
name = "database-Server"
if not params.useVMs:
    dnode = request.RawPC(name)
    dnode.hardware_type = params.other_node_type if params.other_node_type else DEFAULT_NODE_TYPE
else:
    dnode = request.XenVM(name)
    if params.exclusiveVMs:
        dnode.exclusive = True
dnode.disk_image = params.other_os_image if params.other_os_image else VANILLA_OS_IMAGES[2]
dnode_iface = dnode.addInterface("dnode_iface")
dnode_iface.addAddress(rspec.IPv4Address("192.168.1.1", "255.255.255.0"))
dnode_lan = request.LAN("dnode_lan")
if params.sameSwitch:
    dnode_lan.setNoInterSwitchLinks()
if params.bestEffort:
    dnode_lan.best_effort = True
elif params.linkSpeed > 0:
    dnode_lan.bandwidth = params.linkSpeed
dnode_lan.addInterface(dnode_iface)

if params.tempFileSystemSize > 0 or params.tempFileSystemMax:
    bs = dnode.Blockstore(name + "-bs", params.tempFileSystemMount)
    if params.tempFileSystemMax:
        bs.size = "0GB"
    else:
        bs.size = str(params.tempFileSystemSize) + "GB"
    bs.placement = "any"
if params.startVNC:
    dnode.startVNC()

# Create primary and secondary users
for idx, site in enumerate(params.cbrs_radio_sites):
    if idx == 0:
        name = "Primary-User-{}".format(site.device.split("-")[1])
        if not params.useVMs:
            node = request.RawPC(name)
        else:
            node = request.XenVM(name)
            if params.exclusiveVMs:
                node.exclusive = True
        iface = node.addInterface("pnode_iface")
        iface.addAddress(rspec.IPv4Address("192.168.1.2", "255.255.255.0"))
        dnode_lan.addInterface(iface)
    else:
        name = "Secondary-User-{}".format(site.device.split("-")[1])
        if not params.useVMs:
            node = request.RawPC(name)
        else:
            node = request.XenVM(name)
            if params.exclusiveVMs:
                node.exclusive = True
        iface = node.addInterface("snode_iface")
        iface.addAddress(rspec.IPv4Address("192.168.1."+str(idx+2), "255.255.255.0"))
        dnode_lan.addInterface(iface)

    node.hardware_type = params.gnuradio_node_type if params.gnuradio_node_type else DEFAULT_NODE_TYPE
    node.disk_image = params.gnuradio_os_image if params.gnuradio_os_image else GNURADIO_DISK_IMAGE
    if params.tempFileSystemSize > 0 or params.tempFileSystemMax:
        bs = node.Blockstore(name + "-bs", params.tempFileSystemMount)
        if params.tempFileSystemMax:
            bs.size = "0GB"
        else:
            bs.size = str(params.tempFileSystemSize) + "GB"
        bs.placement = "any"
    if params.startVNC:
        node.startVNC()

    # Add SDR
    node_radio_if = node.addInterface("usrp_if")
    node_radio_if.addAddress(pg.IPv4Address("192.168.40.1", "255.255.255.0"))

    radio_link = request.Link("radio-link-{}".format(idx))
    radio_link.bandwidth = 10*1000*1000
    radio_link.addInterface(node_radio_if)

    radio = request.RawPC("{}-sdr".format(site.device))
    radio.component_id = site.device
    radio.component_manager_id = COMP_MANAGER_ID
    radio_link.addNode(radio)

# Request frequency range(s)
for frange in params.freq_ranges:
    request.requestSpectrum(frange.freq_min, frange.freq_max, 0)

# Request ed1+B210 radio resources on all ME units (shuttles).
if params.alloc_shuttles:
    allroutes = request.requestAllRoutes()
    allroutes.disk_image = GNURADIO_DISK_IMAGE

if params.tempFileSystemSize < 0 or params.tempFileSystemSize > 200:
    pc.reportError(portal.ParameterError("Please specify a size greater then zero and " +
                                         "less then 200GB", ["tempFileSystemSize"]))

pc.printRequestRSpec(request)
