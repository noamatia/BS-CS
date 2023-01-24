package bgu.spl.mics.application.passiveObjects;
import java.util.List;

/**
 * Passive data-object representing a delivery vehicle of the store.
 * You must not alter any of the given public methods of this class.
 * <p>
 * You may add ONLY private fields and methods to this class.
 */
public class Report {

	private String MissionName;
	private String gadgetName;
	private int m;
	private int timeIssued;
	private int QTime;
	private int timeCreated;
	private int Moneypenny;
	private List<String> agentSerialNumbers;
	private List<String> agentNames;

	/**
     * Retrieves the mission name.
     */
	public String getMissionName() {return  this.MissionName;}

	/**
	 * Sets the mission name.
	 */
	public void setMissionName(String missionName) {this.MissionName=missionName; }

	/**
	 * Retrieves the M's id.
	 */
	public int getM() {return this.m;}

	/**
	 * Sets the M's id.
	 */
	public void setM(int m) {this.m=m;}

	/**
	 * Retrieves the Moneypenny's id.
	 */
	public int getMoneypenny() {return this.Moneypenny;}

	/**
	 * Sets the Moneypenny's id.
	 */
	public void setMoneypenny(int moneypenny) {this.Moneypenny=moneypenny;}

	/**
	 * Retrieves the serial numbers of the agents.
	 * <p>
	 * @return The serial numbers of the agents.
	 */
	public List<String> getAgentsSerialNumbers() {return this.agentSerialNumbers;}

	/**
	 * Sets the serial numbers of the agents.
	 */
	public void setAgentsSerialNumbers(List<String> agentsSerialNumbersNumber) { this.agentSerialNumbers=agentsSerialNumbersNumber;}

	/**
	 * Retrieves the agents names.
	 * <p>
	 * @return The agents names.
	 */
	public List<String> getAgentsNames() {return this.agentNames;}

	/**
	 * Sets the agents names.
	 */
	public void setAgentsNames(List<String> agentsNames) {this.agentNames=agentsNames;}

	/**
	 * Retrieves the name of the gadget.
	 * <p>
	 * @return the name of the gadget.
	 */
	public String getGadgetName() { return this.gadgetName; }

	/**
	 * Sets the name of the gadget.
	 */
	public void setGadgetName(String gadgetName) { this.gadgetName=gadgetName;}

	/**
	 * Retrieves the time-tick in which Q Received the GadgetAvailableEvent for that mission.
	 */
	public int getQTime() {return this.QTime;}

	/**
	 * Sets the time-tick in which Q Received the GadgetAvailableEvent for that mission.
	 */
	public void setQTime(int qTime) {this.QTime=qTime;}

	/**
	 * Retrieves the time when the mission was sent by an Intelligence Publisher.
	 */
	public int getTimeIssued() {return this.timeIssued;}

	/**
	 * Sets the time when the mission was sent by an Intelligence Publisher.
	 */
	public void setTimeIssued(int timeIssued) {this.timeIssued=timeIssued;}

	/**
	 * Retrieves the time-tick when the report has been created.
	 */
	public int getTimeCreated() {return this.timeCreated;}

	/**
	 * Sets the time-tick when the report has been created.
	 */
	public void setTimeCreated(int timeCreated) {this.timeCreated=timeCreated;}
}
