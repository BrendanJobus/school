import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetSocketAddress;

public class Receiver extends Node
{
	static final int HEADER_LENGTH = 20;
	
	static final int TYPE_POS = 0;
	static final byte TYPE_SYN = 0;
	static final byte TYPE_CONTROLLER_HELLO = 1;
	static final byte TYPE_MSG = 2;
	static final byte TYPE_MAP_REQ = 3;
	static final byte TYPE_FEATURE_REQ = 4;
	static final byte TYPE_NEW_MAP = 5;
	static final byte TYPE_SYN_ACK = 6;
	static final byte TYPE_ACK = 10;
	
	static final int SRC_POS = 1;
	static final int DST_POS = 6;
	static final int LENGTH_POS = 11;
	
	Terminal terminal;
	InetSocketAddress dstAddress;
	int srcPort;
	
	Receiver(Terminal terminal, int port, int routerPort)
	{
		try
		{
			this.terminal = terminal;
			socket = new DatagramSocket(port);
			this.dstAddress = new InetSocketAddress("localhost", routerPort);
			this.srcPort = port;
			listener.go();
		}
		catch(java.lang.Exception e)
		{
			e.printStackTrace();
		}
	}
	
	public synchronized void onReceipt(DatagramPacket packet)
	{
		try
		{
			byte[] data = packet.getData();
			switch(data[TYPE_POS])
			{
			case TYPE_SYN:
				System.out.println("Yay!!!");
				
				DatagramPacket newPacket;
				byte[] newData;
				byte[] dstPort = new byte[DST_POS - SRC_POS];
				byte[] srcPort = new byte[DST_POS - SRC_POS];
				srcPort = Integer.toString(this.srcPort).getBytes();
				System.arraycopy(data, SRC_POS, dstPort, 0, dstPort.length);
				
				newData = new byte[HEADER_LENGTH];
				newData[TYPE_POS] = TYPE_SYN_ACK;
				System.arraycopy(dstPort, 0, newData, DST_POS, dstPort.length);
				System.arraycopy(srcPort, 0, newData, SRC_POS, srcPort.length);
				
				newPacket = new DatagramPacket(newData, newData.length);
				newPacket.setSocketAddress(dstAddress);
				socket.send(newPacket);
				break;
			case TYPE_MSG:
				System.out.println("Message received");
				byte[] messageInBytes = new byte[data[LENGTH_POS]];
				System.arraycopy(data, HEADER_LENGTH, messageInBytes, 0, messageInBytes.length);
				String message = new String(messageInBytes);
				terminal.println(message);
				dstPort = new byte[DST_POS - SRC_POS];
				System.arraycopy(data, SRC_POS, dstPort, 0, DST_POS - SRC_POS);
				sendAck(dstPort);
				break;
			default:
				break;
			}
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}
	}
	
	public synchronized void sendAck(byte[] dstPort)
	{
		byte[] data = new byte[HEADER_LENGTH];
		DatagramPacket packet = null;
		
		data[TYPE_POS] = TYPE_ACK;
		byte[] src = new String(Integer.toString(srcPort)).getBytes();
		System.arraycopy(src, 0, data, SRC_POS, DST_POS - SRC_POS);
		System.arraycopy(dstPort, 0, data, DST_POS, dstPort.length);
		
		packet = new DatagramPacket(data, data.length);
		packet.setSocketAddress(dstAddress);
		try {
			socket.send(packet);
		} 
		catch (IOException e) 
		{
			e.printStackTrace();
		}
	}
	
	public synchronized void run()
	{
		try
		{
			
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}
	}
}
