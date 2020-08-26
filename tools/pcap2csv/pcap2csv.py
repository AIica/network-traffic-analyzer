import re
from subprocess import Popen, PIPE
from typing import List, Union, Pattern, Dict, Any

import dpkt
import numpy as np
from dpkt.ethernet import Ethernet
from numpy import array as ndarray
from tqdm import tqdm


class Pcap2Csv:
    """
        Class for convert .pcap file to csv file
    """

    FEATURES = [
        "proto",
        "subproto",
        "bulk0",
        "bulk1",
        "bulk2",
        "bulk3",
        "client_packet0",
        "client_packet1",
        "server_packet0",
        "server_packet1",
        "client_bulksize_avg",
        "client_bulksize_dev",
        "server_bulksize_avg",
        "server_bulksize_dev",
        "client_packetsize_avg",
        "client_packetsize_dev",
        "server_packetsize_avg",
        "server_packetsize_dev",
        "client_packets_per_bulk",
        "server_packets_per_bulk",
        "client_effeciency",
        "server_efficiency",
        "byte_ratio",
        "payload_ratio",
        "packet_ratio",
        "client_bytes",
        "client_payload",
        "client_packets",
        "client_bulks",
        "server_bytes",
        "server_payload",
        "server_packets",
        "server_bulks",
        "is_tcp"
    ]

    def __init__(self, strip) -> None:
        """Initialize class variables"""

        self.__strip = strip

    def parse_flows(self, pcap: str) -> Dict:
        """Parse flows for pcap file using ndpiReader and dpkt readers"""

        raw_data = self.__read_pcap(pcap)
        regex = self.__build_regular_exp()
        apps, flows = self.__parse_dpkt_reader(**self.__parse_ndpi_reader_info(regex, raw_data), pcap=pcap)
        return self.__parse_flow(apps, flows)

    @staticmethod
    def __read_pcap(pcap: str) -> Union[str, bytes]:
        """Read pcap file using 'ndpiReader' compiled for linux"""

        pipe = Popen(["./ndpiReader", "-i", pcap, "-v2"], stdout=PIPE)
        return pipe.communicate()[0].decode("utf-8")

    def __parse_ndpi_reader_info(self, regex: Pattern[str], raw_data: Union[str, bytes]) -> (List, List):
        """Parse pcap file using ndpiReader"""

        apps = {}
        flows = {}
        for captures in re.findall(regex, raw_data):
            transport_proto, ip1, port1, ip2, port2, app_proto = captures
            ip1 = self.__ip2string(ip1)
            ip2 = self.__ip2string(ip2)
            port1 = int(port1)
            port2 = int(port2)
            key = (transport_proto.lower(),
                   frozenset(((ip1, port1), (ip2, port2))))
            flows[key] = []
            apps[key] = app_proto.split(".")
            if len(apps[key]) == 1:
                apps[key].append(None)
        return dict(apps=apps, flows=flows)

    @staticmethod
    def __parse_dpkt_reader(apps: List, flows: List, pcap: str) -> (List, List):
        """Parse pcap file using dpkt reader"""

        for ts, raw in tqdm(dpkt.pcap.Reader(open(pcap, "rb"))):
            eth = dpkt.ethernet.Ethernet(raw)
            ip = eth.data
            if not isinstance(ip, dpkt.ip.IP):
                continue
            seg = ip.data
            if isinstance(seg, dpkt.tcp.TCP):
                transp_proto = "tcp"
            elif isinstance(seg, dpkt.udp.UDP):
                transp_proto = "udp"
            else:
                continue
            key = (transp_proto, frozenset(((ip.src, seg.sport),
                                            (ip.dst, seg.dport))))
            try:
                assert key in flows
                flows[key].append(eth)
            except AssertionError:
                print(repr(ip.src))
        return apps, flows

    def __parse_flow(self, apps: List, flows: List) -> Dict:
        """Parse flow to dictionary"""

        processed_flows: Dict = dict({feature: [] for feature in self.FEATURES})
        for key, flow in tqdm(flows.items()):
            proto = apps[key][0]
            subproto = apps[key][1]
            stats = self.__forge_flow_stats(flow, self.__strip)
            if stats:
                stats.update({"proto": proto, "subproto": subproto if subproto != '' else 'Unknown'})
                for feature in self.FEATURES:
                    processed_flows[feature].append(stats[feature])
        return processed_flows

    @staticmethod
    def __forge_flow_stats(flow: Ethernet, strip: int) -> Union[
        list, Dict[Union[str, Any], Union[Union[int, ndarray, float], Any]]]:
        """Get flow stats by documentation of dpkt"""

        ip = flow[0].data
        seg = ip.data
        if isinstance(seg, dpkt.tcp.TCP):
            # Check SYN Flag
            try:
                seg2 = flow[1].data.data
            except IndexError:
                return list()
            if not (seg.flags & dpkt.tcp.TH_SYN and seg2.flags & dpkt.tcp.TH_SYN):
                return list()
            proto = "tcp"
            flow = flow[3:]  # Remove tcp handshake
        elif isinstance(seg, dpkt.udp.UDP):
            proto = "udp"
        else:
            raise ValueError("Unknown transport protocol: `{}`".format(
                seg.__class__.__name__))

        if strip > 0:
            flow = flow[:strip]

        client = (ip.src, seg.sport)
        server = (ip.dst, seg.dport)

        client_bulks = []
        server_bulks = []
        client_packets = []
        server_packets = []

        cur_bulk_size = 0
        cur_bulk_owner = "client"
        client_fin = False
        server_fin = False
        for eth in flow:
            ip = eth.data
            seg = ip.data
            if (ip.src, seg.sport) == client:
                if client_fin:
                    continue
                if proto == "tcp":
                    client_fin = bool(seg.flags & dpkt.tcp.TH_FIN)
                client_packets.append(len(seg))
                if cur_bulk_owner == "client":
                    cur_bulk_size += len(seg.data)
                elif len(seg.data) > 0:
                    server_bulks.append(cur_bulk_size)
                    cur_bulk_owner = "client"
                    cur_bulk_size = len(seg.data)
            elif (ip.src, seg.sport) == server:
                if server_fin:
                    continue
                if proto == "tcp":
                    server_fin = bool(seg.flags & dpkt.tcp.TH_FIN)
                server_packets.append(len(seg))
                if cur_bulk_owner == "server":
                    cur_bulk_size += len(seg.data)
                elif len(seg.data) > 0:
                    client_bulks.append(cur_bulk_size)
                    cur_bulk_owner = "server"
                    cur_bulk_size = len(seg.data)
            else:
                raise ValueError("There is more than one flow here!")

        if cur_bulk_owner == "client":
            client_bulks.append(cur_bulk_size)
        else:
            server_bulks.append(cur_bulk_size)

        stats = {
            "bulk0": client_bulks[0] if len(client_bulks) > 0 else 0,
            "bulk1": server_bulks[0] if len(server_bulks) > 0 else 0,
            "bulk2": client_bulks[1] if len(client_bulks) > 1 else 0,
            "bulk3": server_bulks[1] if len(server_bulks) > 1 else 0,
            "client_packet0": client_packets[0] if len(client_packets) > 0 else 0,
            "client_packet1": client_packets[1] if len(client_packets) > 1 else 0,
            "server_packet0": server_packets[0] if len(server_packets) > 0 else 0,
            "server_packet1": server_packets[1] if len(server_packets) > 1 else 0,
        }

        if client_bulks and client_bulks[0] == 0:
            client_bulks = client_bulks[1:]

        if not client_bulks or not server_bulks:
            return list()

        stats.update({
            "client_bulksize_avg": np.mean(client_bulks),
            "client_bulksize_dev": np.std(client_bulks),
            "server_bulksize_avg": np.mean(server_bulks),
            "server_bulksize_dev": np.std(server_bulks),
            "client_packetsize_avg": np.mean(client_packets),
            "client_packetsize_dev": np.std(client_packets),
            "server_packetsize_avg": np.mean(server_packets),
            "server_packetsize_dev": np.std(server_packets),
            "client_packets_per_bulk": len(client_packets) / len(client_bulks),
            "server_packets_per_bulk": len(server_packets) / len(server_bulks),
            "client_effeciency": sum(client_bulks) / sum(client_packets),
            "server_efficiency": sum(server_bulks) / sum(server_packets),
            "byte_ratio": sum(client_packets) / sum(server_packets),
            "payload_ratio": sum(client_bulks) / sum(server_bulks),
            "packet_ratio": len(client_packets) / len(server_packets),
            "client_bytes": sum(client_packets),
            "client_payload": sum(client_bulks),
            "client_packets": len(client_packets),
            "client_bulks": len(client_bulks),
            "server_bytes": sum(server_packets),
            "server_payload": sum(server_bulks),
            "server_packets": len(server_packets),
            "server_bulks": len(server_bulks),
            "is_tcp": int(proto == "tcp")
        })

        return stats

    @staticmethod
    def __build_regular_exp() -> Pattern[str]:
        """Build regular expression for pcap file string"""

        return re.compile(
            r'(UDP|TCP) (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d{1,5}) <-> (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,'
            r'3}):(\d{1,5}) \[proto: [\d+\.]*\d+\/(\w+\.?\w+)*\]')

    @staticmethod
    def __ip2string(ips: str) -> bytes:
        """Convert ip 127.0.0.1 to bytes"""
        # python 2
        # return "".join(chr(int(n)) for n in ips.split("."))

        # python 3
        return b"".join(bytes([int(n)]) for n in ips.split("."))
