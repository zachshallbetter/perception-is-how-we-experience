# Sensor Types

### **Sensors Classification**

There are, in fact, many classifications for sensors. We can classify sensors depending on the type of output signal or the physical parameters they measure, and other considerations could be taken, resulting in various ways to classify sensors. However, in this article, I’ll mention a couple of ways to classify sensors. First of which is the type of output signal, and the second one is the physical parameter they measure.

**Sensor’s Output Signals Classification**

| **Analog Output** | **Digital Output** |
| --- | --- |
| The output of these sensors is an analog voltage that you can measure then determine the desired physical parameter using the sensor’s transfer function. It may also be capacitive or resistive or anything analog. | The output of these sensors is digital data that you can read via serial or parallel communication buses (as UART, SPI, I2C, etc). The typical format for the data is demonstrated exactly in the sensor’s datasheet. |
| **Example**: The temperature sensor (specifically LM35) is an analog sensor whose output | **Example**: The accelerometer sensor (ADXL345) is a digital sensor that sends out its output data via the I2C two-wire bus. |

**Sensor’s Physical Parameter Classification**

| Temperature Sensors | Chemical Sensors | Proximity Sensors | Touch Sensors |
| --- | --- | --- | --- |
| Light Sensors | Tilt Sensor | Metal Detectors | Cameras |
| Humidity Sensor | Vibration Sensor | Magnetic Sensor | Color Sensor |
| Current Sensor | Pressure Sensor | Fingerprint Sensor | GPS |
| Motor Speed Sensor | Bending Sensor | PIR Sensor | Position Sensor |
| Lidar Sensor | Ultrasonic Sensor | Gyroscope Sensor | Accelerometer Sensor |
| Digital Compass Sensor | Sound Sensor (Mic.) | IR Sensor | Odometer Sensor |
|  |  |  |  |

**Range of Operation**

The most important factor to consider in a sensor is the operating range. If you’re designing a boiler system that will control some liquid boiling at 500°c, You shouldn’t use a small LM35 sensor that can only read up to 150°c. You should make sure that the sensor meets the range requirement of your application in order to get the right sensor for it.

**Accuracy (Resolution)**

Decide on the required resolution (accuracy) of the sensor your applications need prior to choosing a sensor. For example, a temperature sensor with an accuracy of  1°c will be sufficient for a boiler embedded system’s design. However, the same sensor with the same accuracy may not be sufficient for some critical scientific experiments or devices that require an accuracy of 0.1°c. So, there is a trade-off and you have to make your own decision based on your system’s specifications.

**Interfacing Methods**

As we’ve stated earlier, some sensors are analog and others are digital. Hence, there are different ways to interface and read these sensors using analog input pins of an MCU. Or connect it to a serial bus like UART, SPI, or I2C. You should also decide on the type of interface that your application can handle much more smoothly without problems or running out of serial ports.

**Data Rate (For Digital Sensors)**

Digital sensors can send you readings (data) at a rate we call the sampling rate. Typically sensors’ rate is defined by ksp/s (kilo samples per second) which is a thousand sample points (readings) in a second. Some sensors can supply up to a few Msp/s. Most of the time, it’s a programmable feature in sensor modules. And you have to check whether this rate of data supplies your MCU with the information it needs to run the algorithm or perform the required calculations flawlessly.

## Sensor Types

### **Temperature Sensors**

**Semiconductor Temperature Sensors** are the devices that come in the form of integrated circuits i.e. ICs hence, popularly known as IC temperature sensors. These are the electronic devices manufactured in an identical fashion to present-day electronic semiconductor devices like microprocessors. More than thousands of devices can be fabricated upon thin silicon wafers. A whole new range of semiconductor temperature sensors is arriving from different manufacturers. However, the most popular ones include AD590 and LM35.

Their design results from the fact that semiconductor diodes have temperature-sensitive voltage vs. current characteristics. When two identical transistors are operated at a constant ratio of collector current densities, the difference in base-emitter voltages is directly proportional to the absolute temperature.

**Negative Temperature Coefficient** (**NTC**) thermistor. The effective operating range is -50 to 250 °C.

**Resistance Temperature Detector** (**RTD**), also known as a resistance thermometer. Platinum RTDs offer a fairly linear output that is highly accurate (0.1 to 1 °C) across -200 to 600 °C. While providing the greatest accuracy, RTDs also tend to be the most expensive of temperature sensors.

**Thermocouple**. A thermocouple is an electrical device consisting of two dissimilar electrical conductors forming electrical junctions at differing temperatures. A thermocouple produces a temperature-dependent voltage as a result of the thermoelectric effect, and this voltage can be interpreted to measure temperature. Thermocouples are a widely used type of temperature sensor.

Thermocouples are nonlinear sensors, requiring conversion when used for temperature control and compensation, typically accomplished using a LUT (lookup table). Accuracy is low, from 0.5 °C to 5 °C. However, they operate across the widest temperature range, from -200 °C to 1750 °C.

### **Pressure Sensors**

Pressure sensors are used for measuring the pressure of gases and liquids. Pressure sensors are used for control and monitoring in thousands of everyday applications. Pressure sensors can also be used to indirectly measure other variables such as fluid/gas flow, speed, water level, and altitude.

**Different Types Of Pressure Sensors**

Pressure sensors can be classified in terms of pressure ranges they measure temperature ranges of operation, and most importantly the type of pressure they measure. Pressure sensors are variously named according to their purpose, but the same technology may be used under different names.

**1 Absolute pressure sensor**

This sensor measures the pressure relative to a perfect vacuum.

**2 Gauge pressure sensor**

This sensor measures the pressure relative to atmospheric pressure. A tire pressure gauge is an example of gauge pressure measurement; when it indicates zero, then the pressure it is measuring is the same as the ambient pressure.

**3 Vacuum pressure sensor**

This term can cause confusion. It may be used to describe a sensor that measures pressures below atmospheric pressure, showing the difference between low-pressure and atmospheric pressure, but it may also be used to describe a sensor that measures absolute pressure relative to a vacuum.

**4 Differential pressure sensor**

This sensor measures the difference between two pressures, one connected to each side of the sensor. Differential pressure sensors are used to measure many properties, such as pressure drops across oil filters or air filters, fluid levels (by comparing the pressure above and below the liquid), or flow rates (by measuring the change in pressure across a restriction). Technically speaking, most pressure sensors are really differential pressure sensors; for example, a gauge pressure sensor is merely a differential pressure sensor in which one side is open to the ambient atmosphere.

### **Chemical Sensors**

A chemical sensor is a self-contained analytical device that can provide information about the chemical composition of its environment, that is, a liquid or a gas phase. The information is provided in the form of a measurable physical signal that is correlated with the concentration of a certain chemical species. Chemical sensors include the gas sensor, methane sensor, hydrogen sensor, Carbon dioxide sensor, etc.

Chemical sensors are widely used in biomedical sensing and diagnosing devices. This type of sensors is used for the diagnostics of gaseous issues such as the concentration of chemicals in human bodies. The monitoring purposes of chemical activities in the body are measured by chemical sensors. The high chemical sensitivity of graphene enables it to be the most desirable component in biomedical devices.

- Chemo resistors
- Chemical field effect transistor
- Fluorescent chloride sensor
- Electrochemical gas sensor
- Non-dispersive infrared sensor
- Potentiometric sensor
- Hydrogen sulfide sensor

### **Humidity Sensor**

The definition of humidity asserts that it’s the percentage of H2O presence in the atmosphere (air), it’s the amount of water vapor existing in the air of a specific area. Humidity measurement in industries is critical because it may affect the business cost of the product and the health and safety of the personnel. Hence, humidity sensing is very important, especially in the control systems for industrial processes and human comfort.

Humidity also can affect and damage metallic equipment by accelerating the chemical corrosion process. In these areas, high humidity is not desirable at all. However, in some applications like farming the humidity is a good thing for several plants and hence it could be actually a good thing.

### **Current Sensors**

A current sensor is an electronic device that can detect electric current DC or AC in a wire, and generates a signal proportional to that current. The generated signal could be analog voltage or current or even a digital output. The generated signal can be then used to display the measured current in an ammeter, or can be stored for further analysis in a data acquisition system, or can be used for the purpose of control.

We typically use current sensors to determine the power consumption in loads, or maybe estimate the torque in DC motors. There are actually endless possibilities for current sensors applications. You can easily interface this sensor with any microcontroller for sure. Here is a picture of the most common current sensor module in the makers’ space.

Hall effect current sensors consist of a core, Hall effect device, and signal conditioning circuitry. The sensor works when the current conductor passes through a magnetically permeable core that concentrates the conductor’s magnetic field. The Hall effect sensor, which is mounted within the core, is at a right angle to the concentrated magnetic field and a constant current (in one plane) excites the Hall sensor. The energized Hall sensor is then exposed to a magnetic field from the core and it produces a potential difference that can be measured and amplified for further processing and monitoring by any microcontroller.

### **Vibration Sensor**

There are typically three different types of sensors that are being used in vibration detection: displacement, velocity, and acceleration. Displacement sensors measure changes in distance between a machine’s rotating element and its stationary housing (frame). Displacement sensors come in the form of a probe that threads into a hole drilled and tapped in the machine’s frame, just above the surface of a rotating shaft. Velocity and acceleration sensors, by contrast, measure the velocity or acceleration of whatever element the sensor is attached to, which is usually some external part of the machine frame.

There are many applications that you can design and make using the vibration sensor and it ranges from small wearables to HVAC systems, other examples include: Automotive industrial applications and motor control and monitoring systems.

Vibration sensing could be also useful for robotics especially mobile robots. You can tell if your robot did crash into something or whether it’s moving over a smooth or rough road. There are many possibilities and different ways to make use of the vibration sensor in your projects.

### **Sound Sensors**

There are many applications where you need to make your system sound-aware. Maybe respond to some sort of sound signals or voice commands or whatever. There are many types of sound sensing devices, in this article, I’ll highlight a couple of them. First of which is the microphones which are the most common devices to pick up sound waves and turns it into an electric signal. And the second type is the piezoelectric elements that also converts the pressure into small electric signals. The sound travels through the air as a wave of pressure and that’s exactly what causes the piezoelectric elements to generate a simial wave but in the form of an electric signal.

**Microphones**

A microphone is the most common sound sensor in the world. It does exist in our smartphones, laptops, and all sound systems whether it’s embedded or standalone. A typical microphone transforms the sound waves into electric signals that could be amplified or sent to a DSP (digital signal processor) for further processing or analysis. There are different types of microphone depending on the construction and the working principle of it.

An electret microphone is a type of electrostatic capacitor-based microphone, which eliminates the need for a polarizing power supply by using a permanently charged material. An electret is a stable dielectric material with a permanently embedded static electric dipole moment (which, due to the high resistance and chemical stability of the material, will not decay for hundreds of years). The name comes from electrostatic and magnet; drawing the analogy to the formation of a magnet by the alignment of magnetic domains in a piece of iron.

**Piezoelectric Elements**

A piezoelectric sensor is a device that uses the piezoelectric effect, to measure changes in pressure, acceleration, temperature, strain, or force by converting them to an electrical charge. Piezoelectric sensors are versatile tools for the measurement of various processes. They are used for quality assurance, process control, and for research and development in many industries.

They have been successfully used in various applications, such as in medical, aerospace, nuclear instrumentation, and as a tilt sensor in consumer electronics or a pressure sensor in the touchpads of mobile phones. In the automotive industry, piezoelectric elements are used to monitor combustion when developing internal combustion engines.

One disadvantage of piezoelectric sensors is that they cannot be used for truly static measurements. A static force results in a fixed amount of charge on the piezoelectric material. In conventional readout electronics, imperfect insulating materials and reduction in internal sensor resistance cause a constant loss of electrons and yields a decreasing signal. Elevated temperatures cause an additional drop in internal resistance and sensitivity. The main effect on the piezoelectric effect is that with increasing pressure loads and temperature, the sensitivity reduces due to twin formation.

### **Magnetic Sensor**

Magnetic sensors are designed and used to detect the magnetic field strength due to the existence of magnets. There are different types of and shapes for magnetic sensors. Some of which are designed to work in contactless applications such as door close indicators. Other types use the hall effect sensing to remotely sense the strength of the magnetic field.

Magnetic sensors are often used for security and military applications such as detection, discrimination, and localization of ferromagnetic and conducting objects, navigation, position tracking, and antitheft systems. There are so many applications that you can create or build using this type of sensors. Just do your research before choosing the right one for your purpose.

### **Light Sensor**

Light sensors are the type of sensors that can detect the light intensity in the surrounding environment. There are many types of light sensors depending on the principle of operation and the type of light energy it can detect (ambient light, IR, laser, etc). In this article, I’ll mention a couple of light sensors, the LDR (light dependent resistance) and the photoconductive cell.

**LDR**

A photoresistor (or light-dependent resistor, LDR, or photo-conductive cell) is a light-controlled variable resistor. The resistance of a photoresistor decreases with increasing incident light intensity. In other words, it exhibits photoconductivity. A photoresistor can be applied in light-sensitive detector circuits and light-activated and dark-activated switching circuits.

A photoresistor is made of a high resistance semiconductor. In the dark, a photoresistor can have a resistance as high as several megohms (MΩ), while in the light, a photoresistor can have a resistance as low as a few hundred ohms. If incident light on a photoresistor exceeds a certain frequency, photons absorbed by the semiconductor give bound electrons enough energy to jump into the conduction band. The resulting free electrons (and their hole partners) conduct electricity, thereby lowering resistance.

**PhotoDetector**

Photodetectors, also called photosensors, are sensors of light or other electromagnetic radiation. A photo-detector has a p–n junction that converts light photons into a current. The absorbed photons make electron-hole pairs in the depletion region. Photodiodes and photo-transistors are a few examples of photo-detectors. Solar cells convert some of the light energy absorbed into electrical energy.

### **Color Sensor**

The color of an object we see in fact is the chromatic light the object reflects in the white light after it absorbs the rest colors. The white color is a mixture of various visible colors, which means it includes each colored light like red (R), green (G), blue (B). Based on the theory of three primary colors, any color is made by mixing the three primary colors (red, green, and blue) in a certain proportion. Thus, knowing the proportion you can get the color of the tested object.

A color sensor does expose the opposing object to white light then it absorbs back the reflected light from the object. This reflected light is passed through filters for red, green, and blue. Then the light intensity of each color is easily found and the sensor outputs these signals to the microcontroller so you can determine the color of an object with your Arduino-based application or any other microcontroller. An example of a color sensor is the TCS3200 like the one in the picture below.

Color sensors are generally used for two specific applications: true color recognition and color mark detection. Sensors used for true color recognition are required to “see” different colors or to distinguish between shades of a specific color. They can be used in either a sorting or matching mode. In sorting mode, the output is activated when the object to be identified is close to the set color. In matching mode, the output is activated when the object to be detected is identical (within tolerance) to the color stored in memory. Color mark detection sensors do not detect the color of the mark, rather, they “see” differences or changes in the mark in contrast with other marks or backgrounds. They are sometimes referred to as contrast sensors.

### **Accelerometer Sensor**

Accelerometers are electronic sensors that measure acceleration, which is the rate of change of the velocity of an object. They measure in meters per second squared (m/s2) or in G-forces (g). A single G-force for us here on planet Earth is equivalent to 9.8 m/s2, but this does vary slightly with elevation (and will be a different value on different planets due to variations in gravitational pull). Accelerometers are useful for sensing vibrations in systems or for orientation applications.

Accelerometers can measure acceleration on one, two, or three axes. 3-axis units are becoming more common as the cost of development for them decreases. Generally, accelerometers contain capacitive plates internally. Some of these are fixed, while others are attached to minuscule springs that move internally as acceleration forces act upon the sensor. As these plates move in relation to each other, the capacitance between them changes. From these changes in capacitance, the acceleration can be determined.

### **Gyroscope Sensor**

Gyroscope sensors, also known as angular rate sensors or angular velocity sensors, are devices that sense angular velocity. The angular velocity is the change in the rotational angle per unit of time. Angular velocity is generally expressed in deg/s (degrees per second).

Applications of gyroscopes include inertial navigation systems, such as in the Hubble Telescope, or inside the steel hull of a submerged submarine. Due to their precision, gyroscopes are also used to maintain direction in tunnel mining. Gyroscopes can be used to construct gyrocompasses, which complement or replace magnetic compasses (in ships, aircraft, and spacecraft, vehicles in general), to assist in stability (bicycles, motorcycles, and ships) or be used as part of an inertial guidance system. MEMS gyroscopes are popular in some consumer electronics, such as smartphones. MEMS gyroscopes are used in automotive roll-over prevention and airbag systems, image stabilization, and have many other potential applications.

Inexpensive vibrating structure microelectromechanical systems (MEMS) gyroscopes have become widely available. These are packaged similarly to other integrated circuits and may provide either analog or digital outputs. In many cases, a single part includes gyroscopic sensors for multiple axes. Some parts incorporate multiple gyroscopes and accelerometers (or multiple-axis gyroscopes and accelerometers), to achieve output that has six full degrees of freedom. These units are called inertial measurement units, or IMUs. An example for the IMU is the very common module MPU6050, which you can easily interface with an Arduino board and use its gyroscope and accelerometer integrated sensors.

### **Digital Compass (Magnetometer)**

A magnetometer is an electronic sensor that measures magnetism, the direction, strength, or relative change of a magnetic field at a particular location. The measurement of the magnetization of a magnetic material (like a ferromagnet) is an example. A compass is one such device, one that measures the direction of an ambient magnetic field, in this case, the Earth’s magnetic field. Many smartphones contain miniaturized microelectromechanical systems (MEMS) magnetometers which are used to detect magnetic field strength and are used as compasses.

A very common digital compass sensor is the HMC5883L. You’ll find a lot of tutorials for this sensor and how to interface it with a microcontroller like Arduino. Applications for the HMC5883L include Mobile Phones, Netbooks, Consumer Electronics, Auto Navigation Systems, and Personal Navigation Devices. the magnitude of Earth’s magnetic fields, from milli-gauss to 8 gausses. Honeywell’s Magnetic Sensors are among the most sensitive and reliable low-field sensors in the industry.

This sensor can be used in robotics to create a feedback closed-loop control system that makes your robot navigate in straight lines without deviation even if it, for some reason, did drift a little bit. The feedback from the compass will maintain the base heading angle (orientation). This applies to drones as well, you can use this sensor, in the same way, to maintain the orientational position of your drone especially if it’s taking photos or videos.

### **IR Sensor**

There are different sensors that depend on the infrared light as a working principle for them. This includes the photo-interrupter that consists of an IR LED and a photodiode that responds to IR light. The most important and common example we see every day is the remote controls for TVs, Airconditioners, etc. The remote control is typically an IR transmitter LED sending bursts of IR light (data) and the receiver has a photodiode that detects these IR messages and responds correspondingly.

An IR LED is a type of diode or simple semiconductor. Electric current is allowed to flow in only one direction in diodes. Emitting infrared rays ranging from 700 nm to 1 mm wavelength. Different IR LEDs may produce infrared light of differing wavelengths, just like different LEDs produce light of different colors. An IR sensor is an electronic device that detects IR radiation falling on it. Proximity sensors (used in touchscreen phones and edge avoiding robots), contrast sensors (used in line following robots) and obstruction counters/sensors (used for counting goods and in burglar alarms) are some applications involving IR sensors.

IR Transmitter LED (on the right) – IR Reciever Diode (on the left)

The IR infrared sensors are being used in a lot of applications in industry, robotics, and much more. You can use it for creating a line follower robot by detecting the line black color. Or you can use it for remotely controlling some electronic devices. Another way to use this sensor is to implement a proximity sensor which we’ll discuss hereafter in this article. There are in fact a whole lot of possibilities and applications that you could implement with IR sensors and transmitters.

### **Proximity Sensors**

A proximity sensor is a sensor able to detect the presence of nearby objects without any physical contact. A proximity sensor often emits an electromagnetic field or a beam of electromagnetic radiation (infrared, for instance), and looks for changes in the field or return signal. The object being sensed is often referred to as the proximity sensor’s target. Different proximity sensor targets demand different sensors. For example, a capacitive proximity sensor or photoelectric sensor might be suitable for a plastic target; an inductive proximity sensor always requires a metal target.

**Different Applications**

Proximity sensors can have high reliability and long functional life because of the absence of mechanical parts and lack of physical contact between the sensor and the sensed object. Proximity sensors are also used in machine vibration monitoring to measure the variation in distance between a shaft and its support bearing.

Proximity sensors are commonly used on mobile devices. When the target is within a nominal range, the device lock screen user interface will appear, thus emerging from what is known as sleep mode. Once the device has awoken from sleep mode, if the proximity sensor’s target is still for an extended period of time, the sensor will then ignore it, and the device will eventually revert into sleep mode. For example, during a telephone call, proximity sensors play a role in detecting (and skipping) accidental touchscreen taps when mobiles are held to the ear.

**Inductive Proximity Sensing**

These non-contact proximity sensors detect ferrous targets, ideally mild steel thicker than one millimeter. They consist of four major components: a ferrite core with coils, an oscillator, a Schmitt trigger, and an output amplifier. The oscillator creates a symmetrical, oscillating magnetic field that radiates from the ferrite core and coil array at the sensing face. When a ferrous target enters this magnetic field, small independent electrical currents called eddy currents are induced on the metal’s surface. This changes the reluctance (natural frequency) of the magnetic circuit, which in turn reduces the oscillation amplitude. As more metal enters the sensing field the oscillation amplitude shrinks and eventually collapses.

**Capacitive Proximity Sensing**

Capacitive proximity sensors can detect both metallic and non-metallic targets in powder, granulate, liquid, and solid form. This, along with their ability to sense through nonferrous materials, makes them ideal for sight glass monitoring, tank liquid level detection, and hopper powder level recognition.

In capacitive sensors, the two conductive plates (at different potentials) are housed in the sensing head and positioned to operate as an open capacitor. Air acts as an insulator; at rest, there is little capacitance between the two plates. Like inductive sensors, these plates are linked to an oscillator, a Schmitt trigger, and an output amplifier. As a target enters the sensing area the capacitance of the two plates increases, causing oscillator amplitude change, in turn changing the Schmitt trigger state, and creating an output signal. The difference between the inductive and capacitive sensors: inductive sensors oscillate until the target is present and capacitive sensors oscillate when the target is present.

### **Metal Detector**

A metal detector is an electronic circuit that acts as a sensor that can tell you if there is metal nearby within its effective range. It has many applications and has been used for ages, in the past, it was the best way to detect landmines. Soldiers did use metal detectors for this purpose. And nowadays, it’s used in gates at airports and prisons to detect if somebody is hiding a metallic weapon or not.

The simplest form of a metal detector consists of an oscillator circuit producing an alternating current that passes through a coil producing an alternating magnetic field. If a piece of electrically conductive metal is close to the coil, eddy currents will be induced in the metal, and this will produce a magnetic field of its own. If another coil is used to measure the magnetic field (acting as a magnetometer), the change in the magnetic field due to the metallic object can be detected.

The size of the coil can limit or optimize the size of the target detected. A very small coil can generally pick up smaller targets better than a larger coil. Conversely, a larger coil can usually detect larger objects from farther away, and sometimes sacrifices being able to detect smaller objects (even up close).

### **Ultrasonic Sensor**

An Ultrasonic sensor is an electronic device that can measure the distance to an object by using sound waves. It measures distance by sending out a sound wave at a specific frequency and listening for that sound wave to bounce back. By recording the elapsed time between the sound wave being generated and the sound wave bouncing back, it is possible to calculate the distance between the sonar sensor and the object. Ultrasonic Sensors are best used in the non-contact detection of { Presence, Level, Position, Distance }.

Since it is known that sound travels through air at about 343 m/s, you can take the time for the sound wave to return and multiply it by 343 meters to find the total round-trip distance of the sound wave. Round-trip means that the sound wave traveled 2 times the distance to the object before it was detected by the sensor; it includes the “trip” from the sonar sensor to the object and the “trip” from the object to the Ultrasonic sensor (after the sound wave bounced off the object). To find the distance to the object, simply divide the round-trip distance in half.

Typical applications for ultrasonic sensors include robot navigation, as well as factory automation. Water-level sensing is another good use and can be accomplished by positioning one sensor above a water surface. Another aquatic application is to use these sensors to “see” the bottom of a body of water, traveling through the water, but reflecting off the bottom surface below. And much more, you can just search for DIY projects with ultrasonic sensors and you’ll be amazed by the dozens of ideas you can actually implement.

### **Lidar Sensor**

LiDAR is a surveying method that measures the distance to a target by illuminating the target with laser light and measuring the reflected light with a sensor. Differences in laser return times and wavelengths can then be used to make digital 3-D representations of the target. It has been widely used in archaeology to map dig-sites and large areas of land, identifying things that couldn’t be seen from the ground. The National Oceanic and Atmospheric Administration in America has used it to map shorelines and the surface of the Earth, and NASA utilized the technology in 1971 when Apollo 15 astronauts mapped the surface of the moon using a laser altimeter.

**Working Principle**

The technique employs ultraviolet (UV), visible, or near-infrared (IR) light to image objects and maps their physical features. Several measurements are taken in quick succession to yield a complex map of the surface at high resolution. LIDAR measures the distance to a target using active sensors which emit an energy source for illumination, instead of relying on sunlight.

It fires rapid pulses of laser light at a surface – anything up to 150,000 pulses a second – usually IR to map land, or water-penetrating green light to measure the seafloor or riverbed. When the light hits the target object, it is reflected back to a sensor that measures the time taken for the pulse to bounce back from the target. The distance to the object is deduced by using the speed of light to calculate the distance traveled accurately. The result is precise three-dimensional information about the target object and its surface characteristics.

**Types & Applications**

Lidar sensors are different in types that depend on the platform itself or the orientation. This sensor is being used in a wide variety of applications including ROVer robots, navigation systems for self-driven cars, car driver assistant, agriculture, archeology, biology, military, and much more. Using this sensor requires a little bit more of research than any other basic sensors and fortunately, you can interface it with microcontrollers such as Arduino which will accelerate the development and testing process for your project or application.

### **Touch Sensor**

Touch sensing technology has evolved greatly in the last decades. We’ll focus our attention on a couple of the most common technologies for touch sensing, the resistive touch sensing (old tech), and the capacitive touch sensing (relatively modern tech).

The capacitive touch switch needs only one electrode to function. The electrode can be placed behind a non-conductive panel such as wood, glass, or plastic. The switch works using body capacitance, a property of the human body that gives it great electrical characteristics. The switch keeps charging and discharging its metal exterior to detect changes in capacitance. When a person touches it, their body increases the capacitance and triggers the switch.

The resistance switch needs two electrodes to be physically in contact with something electrically conductive (for example a finger) to operate. They work by lowering the resistance between two pieces of metal. It is thus much simpler in construction compared to the capacitance switch. Placing one or two fingers across the plates achieves a turn on or closed state. Removing the finger(s) from the metal pieces turns the device off.

There are several applications for touch sensing ranging from small devices UI (user interface) like in cameras and printers to the smartphone which are devices that are completely operated and controlled by a touch screen. And you can build a simple touch sensor with an Arduino board and a metallic touchpad even without getting the sensor module shown in the picture above.

Check out my latest eBook that teaches you how capacitive touch sensing works and how to design your own capacitive touch sensors PADs, sliders, and implement various code techniques in order to make your CapTouch application. It’s now on sale and you can use this 25% discount coupon ESM1K2GV for a limited number of readers, so don’t miss this!

### **PIR Sensor**

A passive infrared sensor (PIR sensor) is an electronic sensor that measures infrared (IR) light radiating from objects in its field of view. They are most often used in PIR-based motion detectors. PIR sensors are commonly used in security alarms and automatic lighting applications. PIR sensors detect general movement but do not give information on who or what moved. For that purpose, an active IR sensor is required.

PIR sensors are commonly called simply PIR. The term passive refers to the fact that PIR devices do not radiate energy for detection purposes. They work entirely by detecting infrared radiation (radiant heat) emitted by or reflected from objects.

A PIR sensor can detect changes in the amount of infrared radiation impinging upon it, which varies depending on the temperature and surface characteristics of the objects in front of the sensor. When an object, such as a person, passes in front of the background, such as a wall, the temperature at that point in the sensor’s field of view will rise from room temperature to body temperature, and then back again.

The sensor converts the resulting change in the incoming infrared radiation into a change in the output voltage, and this triggers the detection. Objects of similar temperature but different surface characteristics may also have a different infrared emission pattern, and thus moving them with respect to the background may trigger the detector as well.

The PIR sensor modules can be used in different applications but the most common things include theft detection and alarm systems and automated security systems.

---

# **Detect Errors in Sensor Data**

With advances in sensor network technology, there is growing confidence in the long-term deployment of IoT sensors. When sensors work as expected, you can reap their optimal benefits and continue normal operations. However, you must be wary of errors affecting your sensor accuracy. The most common sensor data faults include:

- **Short faults:** Single sample spikes or sharp changes in the measured value between two successive sensor readings.
- **Noise faults:** A longer duration or gradually increasing variation in sensor data readings.
- **Constant faults:** Constantly abnormal readings in a large number of successive measurements.

These sensor data errors often occur due to malfunctioning hardware or software. Short-circuited connections, low battery, and physical damage are the most typical hardware problems you can suspect.

Calibration errors can corrupt your sensor codes, resulting in software faults. In this error type, all your readings constantly differ from the true value while maintaining normal patterns.

Calibration errors can corrupt your sensor codes, resulting in software faults. In this error type, all your readings constantly differ from the true value while maintaining normal patterns. Identifying the causes of sensor malfunction can help you detect errors in sensor data. For advanced qualitative detection methods, you can seek the professional help of technicians at Senseware.

### Application Types

- Automation
- Robotics
- Embedded Systems
- Computers
- Smart Cars
- Avionics
- Satellites
- Smart Homes
- Smartphones
- Smart Watches
- Energy plants
- Remote Sensing
- Communications

[Sensor References](Sensor%20Types/Sensor%20References%209622b87a7a234559b9ee16425de9e20b.md)