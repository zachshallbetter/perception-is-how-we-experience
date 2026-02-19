# Protocols

The central component of an IoT ecosystem is connectivity, or specifically communication. IoT has made connection and communication between remote and dispersed entities (devices, sensor nodes, gateway units, cloud servers) possible.

Thanks to IoT and its ability to connect remote devices and things, we now have:

- Efficient systems for [predictive maintenance and management](https://www.embitel.com/predictive-maintenance-solutions) of critical industrial assets
- Over the air (OTA) firmware upgrades for remote device management
- IoT-based telematics platforms for intelligent fleet management and asset tracking

Expertise in IoT communication protocols is essential to understanding IoT systems. These protocols, wired or wireless, not only act as a communication medium, but also provide value-added features to IoT networks. IoT protocols such as Zigbee enable an interference-free, low-power communication, while open-source protocols like Profinet facilitate seamless and faster communication with a wide range of industrial units and equipment.

This blog illustrates 8 IoT communication protocols that have been widely deployed in numerous existing and emerging IoT applications.

**1. Bluetooth**

Bluetooth-compatible IoT sensors are ideal for applications that require short range connectivity and low power communication. Bluetooth protocol has an effective range of 50 to 100 m and supports a data transfer rates up to 1 Mbps.

Recently, IoT developers have shown an inclination toward the low energy version of Bluetooth, Bluetooth Low Energy (BLE), based on the Bluetooth Smart protocol. BLE offers significantly reduced power compared to its predecessor but is not deemed fit for large file transfers.

**2. Zigbee**

Zigbee, based on the IEEE 802.15.4 standard, has emerged as a one of the most widely used communications protocols for embedded applications. Zigbee is used to connect devices in the 10-100 m range and supports a data rate of up to 250 Kbps. As a low-power, low data rate technology, Zigbee is well suited for two-way data transfer between IoT sensors and IoT gateway devices, as well as ad hoc wireless mesh networks.

Through its mesh topology, Zigbee devices can transmit data through intermediate devices over a distance. For IoT applications that require low cost and low power sensor networks in the consumer and industrial segments, Zigbee is an excellent option.

The Zigbee protocol also includes a security framework defined by 128-bit cryptographic keys and cyphering frames.

**3. 6LoWPAN:**

6LoWPAN is a simple wireless mesh technology that makes individual nodes IP-enabled. Its objective is to overcome technical and commercial barriers to connecting all types of devices to the internet.

The 6LoWPan specification also defines encapsulation and header compression mechanisms to exchange of IPv6 packets over IEEE 802.15.4 networks. 6LoWPan integrates security components like Secure Mode and ACL keys, as well as optional TLS. It is a viable option for small form factor devices that require low-power wireless communications.

**4. Wi-Fi:**

Wi-Fi was intended to replace Ethernet and provide easy-to-use short-range wireless connectivity and cross-vendor interoperability through the IEEE 802.11 family of standards.

Wi-Fi is known for faster, larger volume communications, and can transmit over ranges of 50 m using the 2.4 GHz and 5 GHz frequency bands. Its popularity continues to grow thanks to the ubiquity of existing infrastructure.

**5. Cellular:**

IoT applications that require continuous connectivity over very long distances can be designed based on cellular networks like GSM, LTE, EDGE, 3G, 4G and 5G. Cellular networks can communicate with devices over distances of up to 35km.

Cellular technology is beneficial for IoT applications because of the following features:

- Cost-optimized, enhanced coverage for IoT applications through LTE networks like Cat-0 and Cat-1.
- End-to-end security through TLS/DTLS security and on-air encryption of the LTE networks.
- Minimal or zero regulatory regulations. With a cellular network, data can be exchanged at up to 23 dBm.

**6. ModBus**

Modbus protocol is a robust communication standard that is used widely in industrial automation and SCADA systems to send signals from instrumentation, sensors and actuators back to a main controller. Modbus has a broad range of communication protocols that operate over a variety of physical links like RS-485.

Modbus is a serial communications protocol based on a master–slave model at its core.

[](https://data.embeddedcomputing.com/uploads/articles/wp/5839/files-aHViPTYzODY3JmNtZD1pdGVtZWRpdG9yaW1hZ2UmZmlsZW5hbWU9aXRlbWVkaXRvcmltYWdlXzViOTJmNDQzM2I3YzYucG5nJnZlcnNpb249MDAwMCZzaWc9MDlkZjIwMDdmOGM0NWZkODJjZGMyNzk0ODgyNjkyOWU253D)

*The Modbus architecture. Image Source: Researchgate*

The master requests information using a function code that indicates an action that needs to be performed such as:

`read a discrete input; read a first in, first out queue`

The main advantage of using Modbus is that it is a simple, open-source protocol. Development costs for Modbus are low and require minimal hardware design. Additionally, Modbus supports interoperability and compatibility with a range of devices (from different vendors) and systems.

**7. PROFINET**

PROFINET is widely used in industrial automation solutions to connect systems in a manufacturing environment. Based on the IEC 61158 and IEC 61784 standards, PROFINET delivers data at regular time intervals (1 ms or less) without quality loss. It also supports existing IT standards.

PROFINET is highly compatible with fieldbus technologies and can be easily integrated with existing industrial systems. The PROFINET specification defines proxies (proxy addresses) with specified data mapping systems to allow the protocol to communicate with modern IoT protocols.

**8. EtherCAT**

EtherCAT is based on the CANopen protocol and Ethernet and specifically optimized for industrial automation. It allows any standard PC to be used as an EtherCAT master and communicate with EtherCAT slaves using any topology. Together, they can be used to connect all devices on a factory floor at rates of up to 1,000 I/O points in 30 microseconds.

EtherCAT is reliable and delivers fast throughput, as messages are processed with the help of dedicated high-performance hardware before they’re forwarded to the next slave system.