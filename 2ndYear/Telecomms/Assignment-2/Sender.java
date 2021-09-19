import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetSocketAddress;

public class Sender extends Node
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
	
	static final int dstPort = 50006;
	
	Terminal terminal;
	InetSocketAddress dstAddress;
	int srcPort;

	Sender(Terminal terminal, int srcPort, int routerPort)
	{
		try
		{
			this.terminal = terminal;
			this.srcPort = srcPort;
			dstAddress = new InetSocketAddress("localhost", routerPort);
			socket = new DatagramSocket(srcPort);
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
			case TYPE_SYN_ACK:
				System.out.println("Syn successfull");
				this.notify();
				break;
			case TYPE_ACK:
				System.out.println("Message successfully sent");
				terminal.println("message sent");
				this.notify();
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
	
	public synchronized void sendMessage(String message)
	{
		byte[] data = null;
		byte[] buffer = null;
		byte[] srcBytes = null;
		byte[] dstBytes = null;
		DatagramPacket packet = null;
		String src, dst;
		
		System.out.println(message);
		buffer = message.getBytes();
		
		data = new byte[HEADER_LENGTH + buffer.length];
		data[TYPE_POS] = TYPE_MSG;
		src = Integer.toString(srcPort);
		dst = Integer.toString(dstPort);
		srcBytes = src.getBytes();
		dstBytes = dst.getBytes();
		System.arraycopy(srcBytes, 0, data, SRC_POS, srcBytes.length);
		System.arraycopy(dstBytes, 0, data, DST_POS, dstBytes.length);
		
		data[LENGTH_POS] = (byte)buffer.length;
		System.arraycopy(buffer, 0, data, HEADER_LENGTH, buffer.length);
		
		packet = new DatagramPacket(data, data.length);
		packet.setData(data);
		packet.setSocketAddress(dstAddress);
		
			try 
			{
				socket.send(packet);
			} catch (Exception e) {
				e.printStackTrace();
			}
		
		terminal.println("Packet Sent");
	}
	
	public synchronized void sendSyn()
	{

		byte[] data = null;
		byte[] srcBytes = null;
		byte[] dstBytes = null;
		DatagramPacket packet = null;
		String src, dst;
		
		data = new byte[HEADER_LENGTH];
		data[TYPE_POS] = TYPE_SYN;
		src = Integer.toString(srcPort);
		dst = Integer.toString(dstPort);
		srcBytes = src.getBytes();
		dstBytes = dst.getBytes();
		System.arraycopy(srcBytes, 0, data, SRC_POS, srcBytes.length);
		System.arraycopy(dstBytes, 0, data, DST_POS, dstBytes.length);
		
		packet = new DatagramPacket(data, data.length);
		packet.setData(data);
		packet.setSocketAddress(dstAddress);
		
		try 
		{
			socket.send(packet);
		} catch (Exception e) 
		{
			e.printStackTrace();
		}
	}
	
	public synchronized void run()
	{
		try
		{
			String message;
			
			Boolean continueLoop = true;
			while(continueLoop == true)
			{
				message = terminal.read("Message: ");
				sendSyn();
				this.wait();
				sendMessage(message);
				this.wait();
			}
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}
	}
}
