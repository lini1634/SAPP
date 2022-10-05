import pandas as pd 
import pickle
import numpy as np 
import requests
from bs4 import BeautifulSoup
import os 

def read_node(node_path):
    assert os.path.exists(node_path)
    total_hosts = pd.read_csv(node_path,index_col=[0])
    return total_hosts

def vulns_to_vector(x):
    x = eval(x)
    if type(x)==int:
        return x
    else:
        x = list(map(float, x))
        return np.mean(x)

def vulns_label(x):
    if x>5:
        return 1
    else:
        return 0

def make_vulns_to_vector(total_hosts):
    total_hosts["mean_of_cvss"] = total_hosts["vulns"].apply(vulns_to_vector)
    total_hosts["vulns"] =  total_hosts["mean_of_cvss"].apply(vulns_label)
    return total_hosts

def nmap_port_info():
    url = 'https://svn.nmap.org/nmap/nmap-services'
    response = requests.get(url)
    if response.status_code == 200:
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
    else : 
        print(response.status_code)
    port_info = []
    nmapinfo = soup.get_text().split('\n')
    for i in range(22, len(nmapinfo)):
        try:
            data = nmapinfo[i].split('\t') # Service name, portnum/protocol, open-frequency, optional comments
            portnum = data[1].split('/')[0]
            open_frequency = data[2]
            port_info.append(np.array([int(portnum), float(open_frequency)]))
        except:
            pass
    port_info_df = pd.DataFrame(port_info, columns=["port number", "open_frequency"])
    print("nmap services port information: ")
    print(port_info_df.describe())
    open_freq_1 = port_info_df[(port_info_df['open_frequency']>0.01) & (port_info_df['open_frequency']<=0.3)]['port number'].values
    open_freq_2 = port_info_df[port_info_df['open_frequency']>0.3]['port number'].values
    return open_freq_1, open_freq_2

def make_port_to_vector(total_hosts, open_freq_1, open_freq_2, likely_http_ports, likely_ssl_ports, likely_ssh_ports):
    port = total_hosts["port"].values.tolist()
    vulns_port = {}
    other_ports = []
    for p in port: 
        if p in open_freq_2:
            if 'open freq > 0.3' in vulns_port.keys():
                vulns_port['open freq > 0.3'] += 1
            else:
                vulns_port['open freq > 0.3'] = 1
        elif p in open_freq_1:
            if 'open freq > 0.01' in vulns_port.keys():
                vulns_port['open freq > 0.01'] += 1
            else:
                vulns_port['open freq > 0.01'] = 1
        elif p in likely_http_ports:
            if 'likely_http_ports' in vulns_port.keys():  
                vulns_port['likely_http_ports'] += 1
            else: 
                vulns_port['likely_http_ports'] = 1
        elif p in likely_ssl_ports:
            if 'likely_ssl_ports' in vulns_port.keys():  
                vulns_port['likely_ssl_ports'] += 1
            else: 
                vulns_port['likely_ssl_ports'] = 1
        elif p in likely_ssh_ports:
            if 'likely_ssh_ports' in vulns_port.keys():  
                vulns_port['likely_ssh_ports'] += 1
            else: 
                vulns_port['likely_ssh_ports'] = 1
        else:
            if 'other ports' in vulns_port.keys():  
                vulns_port['other ports'] += 1
            else: 
                vulns_port['other ports'] = 1
            other_ports.append(p)
    print("port frequency: ")
    print(vulns_port)

    def port_to_feature(p):
        if p in other_ports:
            return 0 #'vulnerable_port'
            
        elif p in likely_http_ports:
            return 1 #'vulnerable_port'
            
        elif p in likely_ssl_ports:
            return 2 #'vulnerable_port'

        elif p in likely_ssh_ports:
            return 3 #'vulnerable_port'

        elif p in open_freq_1:
            return 4 #'well_known_port'

        elif p in open_freq_2:
            return 5 #'registered_port'

    print("[INFO] make_port_to_vector")
    total_hosts["port"] = total_hosts["port"].apply(port_to_feature)
    return total_hosts

def make_transport_to_vector(total_hosts):
    print("[INFO] make_transport_to_vector..")
    def transport_to_feature(p):
        if p == 'tcp':
            return 0 #'tcp'
        elif p == 'udp':
            return 1 #'udp'
    total_hosts["transport"] = total_hosts["transport"].apply(transport_to_feature)
    return total_hosts

def make_tags_to_vector(total_hosts):
    print("[INFO] make_tags_to_vector..")
    tag_type_list = ['cloud', 'vpn', 'database', 'devops', 'honeypot', 'self-signed', 'etc']
    for tag_type in tag_type_list:
        def tags_to_feature(p):
            if tag_type == 'etc':
                if type(p) == float:
                    return 0
                etc_list = ['cdn', 'starttls', 'iot', 'compromised','cryptocurrency', 'videogame']
                for e in etc_list:
                    if e in p:
                        return 1
                else:
                    return 0
            else:
                if type(p) == float:
                    return 0
                if tag_type in p:
                    return 1
                else:
                    return 0
        total_hosts[tag_type] = total_hosts["tags"].apply(tags_to_feature)
        print(tag_type, ": ", len(total_hosts[total_hosts[tag_type]==1]))
    total_hosts = total_hosts.drop("tags", axis=1)
    return total_hosts

def make_os_to_vector(total_hosts):
    red_hat_linux = ['Red-Hat/Linux', 'Red-Hat/Linux Enterprise Linux']
    cent_os = ['cenos', 'CentOS']
    linux = ['Linux/SuSE', 'Linux/SUSE']

    def os_to_feature(value):
        if type(value) == float:
            return 4 #nothing
        
        idx_os_start = value.find('(')
        idx_os_end = value.find(')')

        second_idx = value.find(')', idx_os_end + 1)
        if idx_os_end >= 0 and second_idx != -1:
            idx_os_end = second_idx

        os_info = value[idx_os_start: idx_os_end+1]
        os_info = os_info.replace('Redhat', 'Red Hat')
        os_info = os_info.replace('centos', 'CentOS')
        os_info = os_info.replace('Linux/SuSE', 'Linux/SUSE')
        os_info = os_info.replace('(Unix)  (Red Hat/Linux)', '(Unix)  (Red-Hat/Linux)')
        os_info = os_info.replace('Red Hat Linux', 'Red-Hat/Linux')
        os_info = os_info.replace('Red Hat', 'Red-Hat/Linux')
        if '(' in os_info:
            os_info = os_info[1:-1]

        if ')  (' in os_info:
            os_1, os_info = os_info.split(')  (')
        if os_info in red_hat_linux:
            return 0 # ['Red-Hat/Linux', 'Red-Hat/Linux Enterprise Linux']
        elif os_info in cent_os:
            return 1 # [centOS]
        elif os_info in linux:
            return 2 # linux
        else:
            return 3 # etc.
        
    print("[INFO] make_os_to_vector..")
    total_hosts['info_os'] = total_hosts["info"].apply(os_to_feature)
    return total_hosts

def main(seed, num_of_nodes):
    node_path = f"../graph_data/DAG_nodes_{seed}_{num_of_nodes}.csv"
    likely_http_ports = [80, 443, 631, 7080, 8080, 8443, 8088, 5800, 3872, 8180, 8000]
    likely_ssl_ports = [261, 271, 324, 443, 465, 563, 585, 636, 853, 989, 990, 992, 993, 994, 995, 2221, 2252, 2376, 3269, 3389, 4433, 4911, 5061, 5986, 6679, 6697, 8443, 9001, 8883]
    likely_ssh_ports = [22, 2222, 55554, 666, 22222, 2382, 830]
    total_hosts = read_node(node_path)
    total_hosts = make_vulns_to_vector(total_hosts)
    open_freq_1, open_freq_2 = nmap_port_info()
    total_hosts = make_port_to_vector(total_hosts, open_freq_1, open_freq_2, likely_http_ports, likely_ssl_ports, likely_ssh_ports)
    total_hosts = make_transport_to_vector(total_hosts)
    total_hosts = make_tags_to_vector(total_hosts)
    total_hosts = make_os_to_vector(total_hosts)
    total_hosts.to_csv(f"../graph_data/DAG_nodes_features_{seed}_{num_of_nodes}.csv")

if __name__ == "__main__":
    seed = 7
    num_of_nodes = 3842
    main(seed, num_of_nodes)