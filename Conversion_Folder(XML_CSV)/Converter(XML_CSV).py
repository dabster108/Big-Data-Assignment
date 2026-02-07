import csv
import os
import xml.etree.ElementTree as ET
from pathlib import Path

def get_xml_text(element, path, ns, default=""):
    """Safely extracts text from a specific XML path with namespaces."""
    found = element.find(path, ns)
    return found.text if found is not None else default

def process_xml_to_csv(input_dir, output_file):
    input_path = Path(input_dir)
    all_rows = []
    
    # TransXChange usually uses this namespace
    ns = {'txc': 'http://www.transxchange.org.uk/'}

    # All attributes for your Analytics & Prediction project
    headers = [
        "FileName", "OperatorName", "LineName", "Direction", "DepartureTime", 
        "JourneyCode", "Sequence", "Activity", "TimingStatus", "RunTime",
        "FromStopRef", "FromStopName", "FromLat", "FromLon",
        "ToStopRef", "ToStopName", "ToLat", "ToLon",
        "SchoolOrgName", "OperatingDates"
    ]

    print(f"Processing XML files in: {input_dir}")

    for xml_file in input_path.glob("*.xml"):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
        except Exception as e:
            print(f"Error parsing {xml_file.name}: {e}")
            continue

        # 1. Map Stop Points (Names and Coordinates)
        stop_map = {}
        for stop in root.findall('.//txc:AnnotatedStopPointRef', ns):
            stop_ref = get_xml_text(stop, 'txc:StopPointRef', ns)
            stop_map[stop_ref] = {
                "name": get_xml_text(stop, 'txc:CommonName', ns),
                "lat": get_xml_text(stop, './/txc:Latitude', ns),
                "lon": get_xml_text(stop, './/txc:Longitude', ns)
            }

        # 2. Extract Serviced Organisation (School Dates)
        org_name = get_xml_text(root, './/txc:ServicedOrganisation/txc:Name', ns)
        date_ranges = []
        for dr in root.findall('.//txc:ServicedOrganisation//txc:DateRange', ns):
            start = get_xml_text(dr, 'txc:StartDate', ns)
            end = get_xml_text(dr, 'txc:EndDate', ns)
            date_ranges.append(f"{start}/{end}")
        date_str = " | ".join(date_ranges)

        # 3. Global Service Info
        line_name = get_xml_text(root, './/txc:Line/txc:LineName', ns)
        op_name = get_xml_text(root, './/txc:OperatorShortName', ns)

        # 4. Map Journey Pattern Sections (The "Legs" of the trip)
        sections_dict = {}
        for section in root.findall('.//txc:JourneyPatternSection', ns):
            sec_id = section.get('id')
            links = []
            for link in section.findall('txc:JourneyPatternTimingLink', ns):
                links.append({
                    "from_ref": get_xml_text(link, 'txc:From/txc:StopPointRef', ns),
                    "to_ref": get_xml_text(link, 'txc:To/txc:StopPointRef', ns),
                    "runtime": get_xml_text(link, 'txc:RunTime', ns),
                    "seq": link.find('txc:From', ns).get('SequenceNumber'),
                    "activity": get_xml_text(link, 'txc:From/txc:Activity', ns),
                    "status": get_xml_text(link, 'txc:From/txc:TimingStatus', ns)
                })
            sections_dict[sec_id] = links

        # 5. Map Patterns to Sections
        pattern_map = {}
        for jp in root.findall('.//txc:JourneyPattern', ns):
            jp_id = jp.get('id')
            pattern_map[jp_id] = {
                "dir": get_xml_text(jp, 'txc:Direction', ns),
                "sec_ref": get_xml_text(jp, 'txc:JourneyPatternSectionRefs', ns)
            }

        # 6. Process Vehicle Journeys (The Actual Rows)
        for vj in root.findall('.//txc:VehicleJourney', ns):
            dep_time = get_xml_text(vj, 'txc:DepartureTime', ns)
            j_code = get_xml_text(vj, './/txc:JourneyCode', ns)
            jp_ref = get_xml_text(vj, 'txc:JourneyPatternRef', ns)
            
            p_info = pattern_map.get(jp_ref, {})
            links = sections_dict.get(p_info.get("sec_ref"), [])

            for link in links:
                f_ref = link["from_ref"]
                t_ref = link["to_ref"]
                
                all_rows.append({
                    "FileName": xml_file.name,
                    "OperatorName": op_name,
                    "LineName": line_name,
                    "Direction": p_info.get("dir"),
                    "DepartureTime": dep_time,
                    "JourneyCode": j_code,
                    "Sequence": link["seq"],
                    "Activity": link["activity"],
                    "TimingStatus": link["status"],
                    "RunTime": link["runtime"],
                    "FromStopRef": f_ref,
                    "FromStopName": stop_map.get(f_ref, {}).get("name", "Unknown"),
                    "FromLat": stop_map.get(f_ref, {}).get("lat"),
                    "FromLon": stop_map.get(f_ref, {}).get("lon"),
                    "ToStopRef": t_ref,
                    "ToStopName": stop_map.get(t_ref, {}).get("name", "Unknown"),
                    "ToLat": stop_map.get(t_ref, {}).get("lat"),
                    "ToLon": stop_map.get(t_ref, {}).get("lon"),
                    "SchoolOrgName": org_name,
                    "OperatingDates": date_str
                })

    # Save to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(all_rows)
    
    print(f"Successfully processed {len(all_rows)} rows into {output_file}")

# --- EXECUTE ---
xml_folder = "/Users/dikshanta/Downloads/bodds_archive_20260120_JP7n6KU/Falcon Buses_237/21332_106265_2026-01-02_16-02-13_current"
output_csv = "final_master_bus_data.csv"

process_xml_to_csv(xml_folder, output_csv)