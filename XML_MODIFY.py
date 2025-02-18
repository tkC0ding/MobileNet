import xml.etree.ElementTree as ET

# Load the XML file
xml_file = "annotations/data_1.xml"  # Change this to your file path
tree = ET.parse(xml_file)
root = tree.getroot()

# List of prefixes to remove
prefixes = ["data_1/", "data_2/", "data_3/", "data_4/", "data_5/"]

# Iterate over all image tags
for image in root.findall(".//image"):
    name = image.get("name")
    if name:
        for prefix in prefixes:
            if name.startswith(prefix):
                new_name = name.replace(prefix, "", 1)  # Remove the matched prefix
                image.set("name", new_name)
                break  # Stop checking other prefixes once a match is found

# Save the modified XML
tree.write("modified_annotations.xml", encoding="utf-8", xml_declaration=True)

print("XML file updated successfully!")
