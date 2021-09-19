import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetSocketAddress;
import java.net.SocketAddress;

public class reader extends Thread
{
	static final int HEADER_LENGTH = 2; // Fixed length of the header
	static final int TYPE_POS = 0; // Position of the type within the header

	static final byte TYPE_UNKNOWN = 0;

	static final byte TYPE_MAKE_CONTACT = 1; // Indicating a string payload
	static final int LENGTH_POS = 1;
	static final int PORT_POS = 2;
	static final int PORT_LENGTH = 5;
	static final int NAME_POS = 7;
	
	static final byte TYPE_WORK_DESCRIPTION = 3; // Indicating the data holds work descriptions
	static final byte TYPE_NUMBER_OF_WORKERS = 4;
	static final int WORK_POS = 2;
	
	static final byte TYPE_WITHDRAW = 5;
	
	static final byte TYPE_SEND_RESULTS = 6;
	
	static final byte TYPE_ACK = 2;   // Indicating an acknowledgement
	static final int ACKCODE_POS = 1; // Position of the acknowledgement type in the header
	static final byte ACK_ALLOK = 10; // Indicating that everything is ok
	
	static final byte PORT_SIZE = 5;

	private final int MAKE_CONTACT = 0;
	private final int WITHDRAW = 1;
	private final int RESULTS = 2;
	private final int FORWARD_WORK = 3;
	private final int SEND_WORK = 4;
	private final int SEND_WORK_TO_WORKERS = 5;
	private final int SEND_ACK = 9;

	int purpose;
	Boolean ackReceived = false;
	InetSocketAddress dstAddress;
	SocketAddress dstSocket;
	DatagramSocket socket;
	DatagramPacket packet;
	int content;
	String info;
	byte[] dataByte;
	
	reader(int purpose, InetSocketAddress dstAddress, DatagramSocket socket, int content, String info)
	{
		this.purpose = purpose;
		this.dstAddress = dstAddress;
		this.socket = socket;
		this.content = content;
		this.info = info;
	}
	
	reader(int purpose, InetSocketAddress dstAddress, DatagramSocket socket, byte[] data)
	{
		this.purpose = purpose;
		this.dstAddress = dstAddress;
		this.socket = socket;
		dataByte = data;
	}
	
	reader(int purpose, InetSocketAddress dstAddress, DatagramSocket socket, int info)
	{
		this.purpose = purpose;
		this.dstAddress = dstAddress;
		this.socket = socket;
		this.content = info;
	}
	
	reader(int purpose, InetSocketAddress dstAddress, DatagramSocket socket, String info)
	{
		this.purpose = purpose;
		this.dstAddress = dstAddress;
		this.socket = socket;
		this.info = info;
	}
	
	reader(int purpose, InetSocketAddress dstAddress, DatagramSocket socket, DatagramPacket packet)
	{
		this.purpose = purpose;
		this.dstAddress = dstAddress;
		this.socket = socket;
		this.packet = packet;
	}
	
	reader(int purpose, SocketAddress dstAddress, DatagramSocket socket, String info)
	{
		this.purpose = purpose;
		this.dstSocket = dstAddress;
		this.socket = socket;
		this.info = info;
	}
	
	reader(int purpose, SocketAddress dstAddress, DatagramSocket socket)
	{
		this.purpose = purpose;
		this.dstSocket = dstAddress;
		this.socket = socket;
	}
	
	public void ackReceived()
	{
		ackReceived = true;
	}
	
	public synchronized void makeContact()
	{
		try
		{
			byte[] data = null;
			byte[] port;
			byte[] name = null;
			DatagramPacket packet = null;
			
			port = (Integer.toString(content)).getBytes();
			name = info.getBytes();
			data = new byte[HEADER_LENGTH + name.length + port.length];
			data[TYPE_POS] = TYPE_MAKE_CONTACT;
			data[LENGTH_POS] = (byte)(name.length + port.length);
			System.arraycopy(port, 0, data, HEADER_LENGTH, port.length);
			System.arraycopy(name, 0, data, NAME_POS, name.length);
			
			packet = new DatagramPacket(data, data.length);
			packet.setSocketAddress(dstAddress);
			while(ackReceived == false)
			{
				socket.send(packet);
				this.wait(200);
			}
		}
		catch(Exception e) {e.printStackTrace();}
	}
	
	public synchronized void withdraw()
	{
		try
		{
			byte[] data;
			byte[] port;
			
			port = (Integer.toString(content)).getBytes();
			data = new byte[HEADER_LENGTH + port.length];
			data[TYPE_POS] = TYPE_WITHDRAW;
			data[LENGTH_POS] = (byte)port.length;
			System.arraycopy(port, 0, data, HEADER_LENGTH, port.length);
			
			DatagramPacket packet;
			packet = new DatagramPacket(data, data.length);
			packet.setSocketAddress(dstAddress);
			while(ackReceived == false) 
			{
				socket.send(packet);
				this.wait(200);
			}
		}
		catch(Exception e) {e.printStackTrace();}
	}
	
	public synchronized void sendResults()
	{
		try
		{
			byte[] data = null;
			byte[] buffer = null;
			DatagramPacket packet= null;

			buffer = info.getBytes();
			data = new byte[HEADER_LENGTH + buffer.length];
			data[TYPE_POS] = TYPE_SEND_RESULTS;
			data[LENGTH_POS] = (byte)buffer.length;
			System.arraycopy(buffer, 0, data, HEADER_LENGTH, buffer.length);

			packet = new DatagramPacket(data, data.length);
			packet.setSocketAddress(dstAddress);
			while(ackReceived == false)
			{
				socket.send(packet);
				this.wait(200);
			}
		}
		catch(Exception e) {e.printStackTrace();}
	}
	
	public synchronized void forwardWork()
	{
		try
		{
			byte[] data;
			byte[] content;
		
			content = new byte[dataByte[LENGTH_POS]];
			System.arraycopy(dataByte, HEADER_LENGTH, content, 0, dataByte[LENGTH_POS]);
			data = new byte[HEADER_LENGTH + content.length];
			data[TYPE_POS] = TYPE_WORK_DESCRIPTION;
			data[LENGTH_POS] = (byte)content.length;
			System.arraycopy(content, 0, data, HEADER_LENGTH, content.length);
			
			DatagramPacket newPacket = new DatagramPacket(data, data.length);
			newPacket.setSocketAddress( dstAddress );
			while(ackReceived == false)
			{
				socket.send(newPacket);
				this.wait(200);
			}
		}
		catch(Exception e) {e.printStackTrace();}
	}
	
	public synchronized void sendWork()
	{
		try
		{
			byte[] data = null;
			byte[] buffer = null;
			DatagramPacket packet = null;
			
			buffer = Integer.toString(content).getBytes();
			data = new byte[HEADER_LENGTH + buffer.length];
			data[TYPE_POS] = TYPE_NUMBER_OF_WORKERS;
			data[LENGTH_POS] = (byte)buffer.length;
			System.arraycopy(buffer, 0, data, HEADER_LENGTH, buffer.length);
			
			packet = new DatagramPacket(data, data.length);
			packet.setSocketAddress(dstAddress);
			while(ackReceived == false)
			{
				socket.send(packet);
				this.wait(200);
			}
			ackReceived = false;
			
			data = null;
			buffer = null;
			packet = null;
			
			buffer = info.getBytes();
			data = new byte[HEADER_LENGTH + buffer.length];
			data[TYPE_POS] = TYPE_WORK_DESCRIPTION;
			data[LENGTH_POS] = (byte)buffer.length;
			System.arraycopy(buffer, 0, data, HEADER_LENGTH, buffer.length);
			
			packet = new DatagramPacket(data, data.length);
			packet.setSocketAddress(dstAddress);
			while(ackReceived == false)
			{
				socket.send(packet);
				this.wait(200);
			}
		}
		catch(Exception e) {e.printStackTrace();}
	}
	
	public synchronized void sendToWorkers()
	{
		try
		{
			byte[] data = null;
			byte[] buffer = null;
			DatagramPacket packet = null;
			
			buffer = info.getBytes();
			data = new byte[HEADER_LENGTH + buffer.length];
			data[TYPE_POS] = TYPE_WORK_DESCRIPTION;
			data[LENGTH_POS] = (byte)buffer.length;
			System.arraycopy(buffer, 0, data, HEADER_LENGTH, buffer.length);
			
			packet = new DatagramPacket(data, data.length);
			packet.setSocketAddress(dstAddress);
			while(ackReceived == false)
			{
				socket.send(packet);
				this.wait(200);
			}
		}
		catch(Exception e) {e.printStackTrace();}
	}
	
	public synchronized void sendAck()
	{
		try
		{
			byte[] data;
			data = new byte[HEADER_LENGTH];
			data[TYPE_POS] = TYPE_ACK;
			data[ACKCODE_POS] = ACK_ALLOK;
			
			DatagramPacket ack;
			ack = new DatagramPacket(data, data.length);
			if(dstAddress != null)
			{
				ack.setSocketAddress( dstAddress );
			}
			else
			{
				ack.setSocketAddress( dstSocket );
			}
			socket.send(ack);
		}
		catch(Exception e) {e.printStackTrace();}
	}
	
	public synchronized void run()
	{
		switch(purpose)
		{
		case MAKE_CONTACT:
			makeContact();
			break;
		case WITHDRAW:
			withdraw();
			break;
		case RESULTS:
			sendResults();
			break;
		case FORWARD_WORK:
			forwardWork();
			break;
		case SEND_WORK:
			sendWork();
			break;
		case SEND_WORK_TO_WORKERS:
			sendToWorkers();
			break;
		case SEND_ACK:
			sendAck();
			break;
		default:
			break;
		}
	}
}
