import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetSocketAddress;
import java.util.Hashtable;

public class Controller extends Node
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
	Hashtable<Integer, Hashtable<Integer, Integer[]>> realFlowtable = new Hashtable<Integer, Hashtable<Integer, Integer[]>>();
	Controller(Terminal terminal, int port, int router1Port, int router2Port, 
			int router3Port, int router4Port, int senderPort, int receiverPort)
	{
		try
		{
			this.terminal = terminal;
			socket = new DatagramSocket(port);
			
			Hashtable<Integer, Integer[]> flowtableValue = new Hashtable<Integer, Integer[]>();
			Integer[] path = {50002, 50003, 50004, 50005};
			flowtableValue.put(50006, path);
			realFlowtable.put(50001, flowtableValue);
			
			flowtableValue = new Hashtable<Integer, Integer[]>();
			Integer[] pathReversed = {50005, 50004, 50003, 50002};
			flowtableValue.put(50001, pathReversed);
			realFlowtable.put(50006, flowtableValue);
			
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
			case TYPE_CONTROLLER_HELLO:
				terminal.println("Hello received");
				sendFeatureRequest(packet);
				break;
			case TYPE_FEATURE_REQ:
				break;
			case TYPE_SYN:
			case TYPE_SYN_ACK:
				System.out.println("Controll Reached");
				byte[] srcPort = new byte[5];
				byte[] dstPort = new byte[5];
				int src, dst;
				System.arraycopy(data, SRC_POS, srcPort, 0, 5);
				System.arraycopy(data, DST_POS, dstPort, 0, 5);
				src = Integer.parseInt(new String(srcPort));
				dst = Integer.parseInt(new String(dstPort));
				if(realFlowtable.containsKey(src))
				{
					Hashtable<Integer, Integer[]> possiblePaths = new Hashtable<Integer, Integer[]>();
					possiblePaths = realFlowtable.get(src);
					
					if(possiblePaths.containsKey(dst))
					{
						Integer[] path = possiblePaths.get(dst);
						
						byte[] newData = null;
						byte[] buffer = null;
						DatagramPacket newPacket = null;
						
						
						for(int i = 0; i < path.length - 1; i++)
						{
							newData = new byte[HEADER_LENGTH + 5];
							buffer = new byte[5];
							newData[TYPE_POS] = TYPE_NEW_MAP;
							newData[LENGTH_POS] = 5;
							System.arraycopy(srcPort, 0, newData, SRC_POS, DST_POS - SRC_POS);
							System.arraycopy(dstPort, 0, newData, DST_POS, DST_POS - SRC_POS);
							buffer = Integer.toString(path[i + 1]).getBytes();
							System.arraycopy(buffer, 0, newData, HEADER_LENGTH, buffer.length);							
							
							newPacket = new DatagramPacket(newData, newData.length);
							newPacket.setSocketAddress(new InetSocketAddress("localhost", path[i]));
							socket.send(newPacket);
							newData = null;
							newPacket = null;
							buffer = null;
						}
						packet.setSocketAddress(new InetSocketAddress("localhost", packet.getPort()));
						socket.send(packet);
					}
				}
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
	
	public synchronized void sendFeatureRequest(DatagramPacket packet)
	{
		try
		{
			DatagramPacket newPacket = null;
			byte[] data = packet.getData();
			
			data[TYPE_POS] = TYPE_FEATURE_REQ;
			newPacket = new DatagramPacket(data, data.length);
			newPacket.setSocketAddress(packet.getSocketAddress());
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
