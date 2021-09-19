import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetSocketAddress;
import java.util.Hashtable;

public class Router extends Node
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
	
	static final int DEFAULT_CONTROLLER_PORT = 50000;
	
	Terminal terminal;
	Hashtable<Integer, Integer> realFlowtable = new Hashtable<Integer, Integer>();
	int leftPort, rightPort, controllerPort;

	Router(Terminal terminal, int port, int leftPort, int rightPort, int controllerPort)
	{
		try
		{
			this.terminal = terminal;
			socket = new DatagramSocket(port);
			this.leftPort = leftPort;
			this.rightPort = rightPort;
			this.controllerPort = controllerPort;
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
			byte[] data;
			data = packet.getData();
			
			switch(data[TYPE_POS])
			{
			case TYPE_FEATURE_REQ:
				terminal.println("Feature Request");
				sendFeatureResponse(packet);
				break;
			case TYPE_SYN:
			case TYPE_SYN_ACK:
			case TYPE_MSG:
			case TYPE_ACK:
				byte[] srcPort = new byte[DST_POS - SRC_POS];
				byte[] dstPort = new byte[DST_POS - SRC_POS];
				System.arraycopy(data, SRC_POS, srcPort, 0, srcPort.length);
				System.arraycopy(data, DST_POS, dstPort, 0, dstPort.length);
								
				System.out.println("Router");
				
				if(packet.getPort() == controllerPort)
				{
					packet.setSocketAddress(new InetSocketAddress("localhost", realFlowtable.get(Integer.parseInt(new String(dstPort)))));
					socket.send(packet);
				}
				else if(Integer.parseInt(new String(dstPort)) == rightPort)
				{
					packet.setSocketAddress(new InetSocketAddress("localhost", rightPort));
					socket.send(packet);
					
				}
				else if(Integer.parseInt(new String(dstPort)) == leftPort)
				{
					packet.setSocketAddress(new InetSocketAddress("localhost", leftPort));
					socket.send(packet);
				}
				else if(realFlowtable.isEmpty())
				{
					packet.setSocketAddress(new InetSocketAddress("localhost", controllerPort));
					socket.send(packet);
				}
				else if(realFlowtable.containsKey(Integer.parseInt(new String(dstPort))))
				{
					packet.setSocketAddress(new InetSocketAddress("localhost", realFlowtable.get(Integer.parseInt(new String(dstPort)))));
					socket.send(packet);
				}
				else
					System.out.println("Uh-oh, error time");
				break;
			case TYPE_NEW_MAP:
				srcPort = new byte[DST_POS - SRC_POS];
				dstPort = new byte[DST_POS - SRC_POS];
				System.arraycopy(data, SRC_POS, srcPort, 0, srcPort.length);
				System.arraycopy(data, DST_POS, dstPort, 0, dstPort.length);
				byte[] nodeToSendTo = new byte[data[LENGTH_POS]];
				System.arraycopy(data, HEADER_LENGTH, nodeToSendTo, 0, nodeToSendTo.length);
				Integer dst = Integer.parseInt(new String(dstPort));
				Integer nextNode = Integer.parseInt(new String(nodeToSendTo));
				realFlowtable.put(dst, nextNode);
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
	
	public synchronized void sendHi()
	{
		try
		{
			byte[] data = null;
			DatagramPacket packet = null;
			
			data = new byte[HEADER_LENGTH];
			data[TYPE_POS] = TYPE_CONTROLLER_HELLO;
			data[LENGTH_POS] = 0;
			
			packet = new DatagramPacket(data, data.length);
			packet.setSocketAddress(new InetSocketAddress("localhost", DEFAULT_CONTROLLER_PORT));
			socket.send(packet);
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}
	}
	
	public synchronized void sendFeatureResponse(DatagramPacket packet)
	{
		try
		{
			byte[] data = null;
			DatagramPacket newPacket = null;
			
			data = packet.getData();
			
			newPacket = new DatagramPacket(data, data.length);
			newPacket.setSocketAddress(new InetSocketAddress("localhost", DEFAULT_CONTROLLER_PORT));
			socket.send(packet);
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}
	}
	
	public synchronized void run()
	{
		try
		{
			terminal.println("Running...");
			sendHi();
			Boolean continueLoop = true;
			while(continueLoop == true)
			{
				this.wait();
			}
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}
	}
}
