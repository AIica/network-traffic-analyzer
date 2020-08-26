import argparse

import pandas as ps
from pcap2csv import Pcap2Csv


def process_pcap(pcap: str, output: str, strip: int) -> None:
    """Process pcap"""

    processor: Pcap2Csv = Pcap2Csv(strip)
    flows = processor.parse_flows(pcap)
    data = ps.DataFrame(flows)
    data.to_csv(output, index=False)


def main() -> None:
    """Main function for process pcap"""

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pcap", help="pcap file")
    parser.add_argument("-o", "--output", help="output csv file", default="main.csv")
    parser.add_argument("-s", "--strip", help="leave only first N datagramms", metavar="N", default=0, type=int)
    argv = parser.parse_args()

    if not argv.pcap:
        raise FileNotFoundError('Pcap was not found')

    process_pcap(**vars(argv))


if __name__ == "__main__":
    main()
