import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetSocketAddress;

public class CommandAndControl extends Node
{
	static final int DEFAULT_SRC_PORT = 50000; // Port of the CommandAndControl
	static final int DEFAULT_DST_PORT = 50001; // Port of the Broker
	static final String DEFAULT_DST_NODE = "localhost";	// Name of the host for the server
	
	// Header Information //
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
	
	static final byte TYPE_ACCEPT_WORK = 5;

	static final byte TYPE_ACK = 2;   // Indicating an acknowledgement
	static final int ACKCODE_POS = 1; // Position of the acknowledgement type in the header
	static final byte ACK_ALLOK = 10; // Indicating that everything is ok
	
	static final byte PORT_SIZE = 5;

	@SuppressWarnings("unused")
	private final int MAKE_CONTACT = 0;
	@SuppressWarnings("unused")
	private final int WITHDRAW = 1;
	@SuppressWarnings("unused")
	private final int RESULTS = 2;
	@SuppressWarnings("unused")
	private final int FORWARD_WORK = 3;
	private final int SEND_WORK = 4;
	@SuppressWarnings("unused")
	private final int SEND_WORK_TO_WORKERS = 5;
	private final int SEND_ACK = 9;

	Terminal terminal;
	InetSocketAddress dstAddress;
	reader reader;
	Thread readerThread;

	
	CommandAndControl(Terminal terminal, String dstHost, int dstPort, int srcPort)
	{
		try 
		{
			this.terminal= terminal;
			dstAddress = new InetSocketAddress(dstHost, dstPort);
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
			case TYPE_ACK:
				reader.ackReceived();
				this.notify();
				break;
			default:
				terminal.println( "Unexpected packet" + packet.toString() );
			}
		}
		catch(Exception e) {e.printStackTrace();}
	}
	
	public synchronized void sendNewWorkDescription(String work, String workerToSendTo)
	{
		try
		{
			int workersWanted = Integer.parseInt(workerToSendTo);
			reader = new reader(SEND_WORK, dstAddress, socket, workersWanted, work);
			readerThread = new Thread( reader );
			readerThread.start();
			terminal.println("Sending work...");
			this.wait();
			this.wait();
			terminal.println("Work sent");
		}
		catch(Exception e) {e.printStackTrace();}
	}
	
	public synchronized void sendAck(DatagramPacket packet)
	{
		try
		{
			reader = new reader(SEND_ACK, dstAddress, socket);
			readerThread = new Thread( reader );
			readerThread.start();
		}
		catch(Exception e) {e.printStackTrace();}
	}
	
	public synchronized void run()
	{	
		try
		{
			String work, sendToWorkers;
			terminal.println("Running...");
			while(true)
			{
				terminal.println("How many workers would you like to send work to?");
				terminal.println("Accepted Inputs: \"all\", \"1\", \"2\", \"3\"....");
				sendToWorkers = terminal.read("Send to how many: ");
				if( sendToWorkers.equalsIgnoreCase("all") )
				{
					work = terminal.read("Work Description: ");
					sendNewWorkDescription( work, "0" );
				}
				else
				{
					work = terminal.read("Work Description: ");
					sendNewWorkDescription( work, sendToWorkers );
				}
			}			
		}
		catch(Exception e) {e.printStackTrace();}
	}
	
	public static void main(String[] args)
	{
		try {
			Terminal terminal = new Terminal("C&C");
			(new CommandAndControl(terminal, DEFAULT_DST_NODE, DEFAULT_DST_PORT, DEFAULT_SRC_PORT)).run();
			terminal.println("Program completed");
		} catch(java.lang.Exception e) {e.printStackTrace();}
	}
}
