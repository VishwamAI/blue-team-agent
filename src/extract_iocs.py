import xml.etree.ElementTree as ET

def extract_iocs(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    iocs = {'urls': [], 'fqdns': [], 'ipv4s': []}
    print("Starting to parse the XML file...")
    for indicator in root.findall('.//stix:Indicator', namespaces={'stix': 'http://stix.mitre.org/stix-1'}):
        observable = indicator.find('.//indicator:Observable', namespaces={'indicator': 'http://stix.mitre.org/Indicator-2'})
        if observable is not None:
            obj = observable.find('.//cybox:Object', namespaces={'cybox': 'http://cybox.mitre.org/cybox-2'})
            if obj is not None:
                properties = obj.find('.//cybox:Properties', namespaces={'cybox': 'http://cybox.mitre.org/cybox-2'})
                if properties is not None:
                    if properties.attrib.get('{http://www.w3.org/2001/XMLSchema-instance}type') == 'URIObj:URIObjectType':
                        value = properties.find('.//URIObj:Value', namespaces={'URIObj': 'http://cybox.mitre.org/objects#URIObject-2'}).text
                        print(f"Found URL: {value}")
                        iocs['urls'].append(value)
                    elif properties.attrib.get('{http://www.w3.org/2001/XMLSchema-instance}type') == 'DomainNameObj:DomainNameObjectType':
                        value = properties.find('.//DomainNameObj:Value', namespaces={'DomainNameObj': 'http://cybox.mitre.org/objects#DomainNameObject-1'}).text
                        print(f"Found FQDN: {value}")
                        iocs['fqdns'].append(value)
                    elif properties.attrib.get('{http://www.w3.org/2001/XMLSchema-instance}type') == 'AddressObj:AddressObjectType':
                        value = properties.find('.//AddressObj:Address_Value', namespaces={'AddressObj': 'http://cybox.mitre.org/objects#AddressObject-2'}).text
                        print(f"Found IPv4: {value}")
                        iocs['ipv4s'].append(value)
    print("Finished parsing the XML file.")
    return iocs

if __name__ == '__main__':
    iocs = extract_iocs('black_basta_iocs.xml')
    print(iocs)
