import socket

server_ip = '0.0.0.0'
server_port = 6969

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((server_ip, server_port))
server_socket.listen(5)
print(f"Listening on {server_ip}:{server_port}")

while True:
    client_socket, addr = server_socket.accept()
    print(f"Connection from {addr}")

    # Receive data from the client
    data = client_socket.recv(1024).decode()
    if data:
        print("Received data:")
        print(data)  # Display full received data

    client_socket.close()
